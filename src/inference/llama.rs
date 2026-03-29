use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend as LlamaInit;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};

use super::InferenceBackend;

// GBNF grammar placeholder — crashes with llama-cpp-2 0.1.140 on ROCm.
// See CLAUDE.md "GBNF grammar sampling crashes" for details.
// const ENGRAM_JSON_GRAMMAR: &str = "root ::= \"hello\"\n";

/// Configuration for the Llama backend.
pub struct LlamaConfig {
    /// Path to the extraction model GGUF (e.g. Qwen3-8B).
    pub generation_model_path: Option<String>,
    /// Path to the embedding model GGUF (e.g. nomic-embed-text).
    pub embedding_model_path: Option<String>,
    /// Number of GPU layers to offload (0 = CPU only).
    pub n_gpu_layers: u32,
    /// Context size for generation.
    pub generation_ctx_size: u32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            generation_model_path: None,
            embedding_model_path: None,
            n_gpu_layers: 99,
            generation_ctx_size: 16384,
        }
    }
}

/// Llama.cpp-based inference backend using GGUF models.
pub struct LlamaBackend {
    _init: LlamaInit,
    generation_model: Option<LlamaModel>,
    embedding_model: Option<LlamaModel>,
    gen_ctx: Option<Mutex<GenerationCtx>>,
    embedding_dim: usize,
}

// SAFETY: LlamaModel is Send+Sync. The context is behind Mutex.
unsafe impl Send for LlamaBackend {}
unsafe impl Sync for LlamaBackend {}

struct GenerationCtx {
    ctx: LlamaContext<'static>,
    ctx_size: u32,
}

impl LlamaBackend {
    pub fn new(config: LlamaConfig) -> Result<Self> {
        // Suppress llama.cpp's verbose log output unless we're at debug level.
        // Must be called before LlamaInit::init().
        let show_llama_logs = tracing::enabled!(tracing::Level::DEBUG);
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(show_llama_logs)
        );

        let init = LlamaInit::init().context("failed to initialize llama.cpp backend")?;

        // Ensure ROCm uses the discrete GPU (device 0 = 7900 XT), not the iGPU
        // SAFETY: called once at init before any threads are spawned
        unsafe { std::env::set_var("HIP_VISIBLE_DEVICES", "0") };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);

        let generation_model = if let Some(path) = &config.generation_model_path {
            tracing::info!("Loading generation model: {path}");
            Some(
                LlamaModel::load_from_file(&init, Path::new(path), &model_params)
                    .map_err(|e| anyhow::anyhow!("failed to load generation model: {e}"))?,
            )
        } else {
            None
        };

        let embedding_model = if let Some(path) = &config.embedding_model_path {
            tracing::info!("Loading embedding model: {path}");
            // Use CPU for embedding model — BERT models have issues with Vulkan in this version
            let cpu_params = LlamaModelParams::default().with_n_gpu_layers(0);
            Some(
                LlamaModel::load_from_file(&init, Path::new(path), &cpu_params)
                    .map_err(|e| anyhow::anyhow!("failed to load embedding model: {e}"))?,
            )
        } else {
            None
        };

        let embedding_dim = embedding_model
            .as_ref()
            .map(|m| m.n_embd() as usize)
            .unwrap_or(768);

        let gen_ctx = if let Some(ref model) = generation_model {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(config.generation_ctx_size))
                .with_n_batch(config.generation_ctx_size)
                .with_embeddings(false);
            // SAFETY: model lives in the same struct as the context.
            let model_ref: &'static LlamaModel = unsafe { &*(model as *const LlamaModel) };
            let ctx = model_ref
                .new_context(&init, ctx_params)
                .map_err(|e| anyhow::anyhow!("failed to create generation context: {e}"))?;
            Some(Mutex::new(GenerationCtx {
                ctx,
                ctx_size: config.generation_ctx_size,
            }))
        } else {
            None
        };

        Ok(Self {
            _init: init,
            generation_model,
            embedding_model,
            gen_ctx,
            embedding_dim,
        })
    }

    /// Create a backend with only the embedding model loaded (for MCP serve).
    pub fn embedding_only(model_path: &str, _n_gpu_layers: u32) -> Result<Self> {
        Self::new(LlamaConfig {
            generation_model_path: None,
            embedding_model_path: Some(model_path.to_string()),
            n_gpu_layers: 0,
            ..Default::default()
        })
    }

    /// Create a backend with both models loaded (for reconcile).
    pub fn full(
        generation_model_path: &str,
        embedding_model_path: &str,
        n_gpu_layers: u32,
    ) -> Result<Self> {
        Self::new(LlamaConfig {
            generation_model_path: Some(generation_model_path.to_string()),
            embedding_model_path: Some(embedding_model_path.to_string()),
            n_gpu_layers,
            ..Default::default()
        })
    }

    /// Generate with GBNF grammar constraining output to valid JSON arrays.
    fn generate_with_grammar(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        let gen_model = self
            .generation_model
            .as_ref()
            .context("generation model not loaded")?;
        let gen_ctx_mutex = self
            .gen_ctx
            .as_ref()
            .context("generation context not available")?;
        let mut gen_state = gen_ctx_mutex.lock().unwrap();

        let tokens = gen_model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        if tokens.len() as u32 >= gen_state.ctx_size {
            bail!(
                "prompt ({} tokens) exceeds context size ({})",
                tokens.len(),
                gen_state.ctx_size
            );
        }

        gen_state.ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        if tokens.len() > 1 {
            for (pos, &token) in tokens[..tokens.len() - 1].iter().enumerate() {
                batch
                    .add(token, pos as i32, &[0], false)
                    .map_err(|e| anyhow::anyhow!("batch add failed: {e}"))?;
            }
        }
        batch
            .add(
                tokens[tokens.len() - 1],
                (tokens.len() - 1) as i32,
                &[0],
                true,
            )
            .map_err(|e| anyhow::anyhow!("batch add failed: {e}"))?;

        gen_state
            .ctx
            .decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

        // Note: GBNF grammar sampling crashes with both Vulkan and ROCm backends
        // in llama-cpp-2 0.1.140 (C++ exception Rust can't catch).
        // JSON validity enforced by the response parser instead.

        // Gemma 3 recommended: temp=1.0, top_k=64, top_p=0.95
        let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
            llama_cpp_2::sampling::LlamaSampler::top_k(64),
            llama_cpp_2::sampling::LlamaSampler::top_p(0.95, 1),
            llama_cpp_2::sampling::LlamaSampler::temp(1.0),
            llama_cpp_2::sampling::LlamaSampler::dist(0),
        ]);

        let mut output = String::new();
        let mut n_decoded = 0u32;
        let mut cur_pos = tokens.len() as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        loop {
            if n_decoded >= max_tokens {
                break;
            }

            let token = sampler.sample(&gen_state.ctx, -1);
            sampler.accept(token);

            if gen_model.is_eog_token(token) {
                break;
            }

            let piece = gen_model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| anyhow::anyhow!("token decode failed: {e}"))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, cur_pos, &[0], true)
                .map_err(|e| anyhow::anyhow!("batch add failed: {e}"))?;
            cur_pos += 1;

            gen_state
                .ctx
                .decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

            n_decoded += 1;
        }

        Ok(output)
    }
}

impl InferenceBackend for LlamaBackend {
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        let gen_model = self
            .generation_model
            .as_ref()
            .context("generation model not loaded")?;
        let gen_ctx_mutex = self
            .gen_ctx
            .as_ref()
            .context("generation context not available")?;
        let mut gen_state = gen_ctx_mutex.lock().unwrap();

        let tokens = gen_model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        if tokens.len() as u32 >= gen_state.ctx_size {
            bail!(
                "prompt ({} tokens) exceeds context size ({})",
                tokens.len(),
                gen_state.ctx_size
            );
        }

        gen_state.ctx.clear_kv_cache();

        // Build batch: logits only on last token
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        if tokens.len() > 1 {
            for (pos, &token) in tokens[..tokens.len() - 1].iter().enumerate() {
                batch
                    .add(token, pos as i32, &[0], false)
                    .map_err(|e| anyhow::anyhow!("batch add failed: {e}"))?;
            }
        }
        batch
            .add(
                tokens[tokens.len() - 1],
                (tokens.len() - 1) as i32,
                &[0],
                true,
            )
            .map_err(|e| anyhow::anyhow!("batch add failed: {e}"))?;

        gen_state
            .ctx
            .decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

        // Gemma 3 recommended: temp=1.0, top_k=64, top_p=0.95
        let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
            llama_cpp_2::sampling::LlamaSampler::top_k(64),
            llama_cpp_2::sampling::LlamaSampler::top_p(0.95, 1),
            llama_cpp_2::sampling::LlamaSampler::temp(1.0),
            llama_cpp_2::sampling::LlamaSampler::dist(0),
        ]);

        let mut output = String::new();
        let mut n_decoded = 0u32;
        let mut cur_pos = tokens.len() as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        loop {
            if n_decoded >= max_tokens {
                break;
            }

            let token = sampler.sample(&gen_state.ctx, -1);
            sampler.accept(token);

            if gen_model.is_eog_token(token) {
                break;
            }

            let piece = gen_model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| anyhow::anyhow!("token decode failed: {e}"))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, cur_pos, &[0], true)
                .map_err(|e| anyhow::anyhow!("batch add failed: {e}"))?;
            cur_pos += 1;

            gen_state
                .ctx
                .decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

            n_decoded += 1;
        }

        Ok(output)
    }

    fn generate_chat(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<String> {
        let gen_model = self
            .generation_model
            .as_ref()
            .context("generation model not loaded")?;

        // Build chat messages and apply the model's template
        let system_msg = LlamaChatMessage::new(
            "system".to_string(),
            system.to_string(),
        ).map_err(|e| anyhow::anyhow!("chat message error: {e}"))?;

        let user_msg = LlamaChatMessage::new(
            "user".to_string(),
            user.to_string(),
        ).map_err(|e| anyhow::anyhow!("chat message error: {e}"))?;

        let template = gen_model
            .chat_template(None)
            .map_err(|e| anyhow::anyhow!("no chat template in model: {e}"))?;

        let formatted = gen_model
            .apply_chat_template(&template, &[system_msg, user_msg], true)
            .map_err(|e| anyhow::anyhow!("chat template failed: {e}"))?;

        // Use grammar-constrained generation for structured JSON output
        self.generate_with_grammar(&formatted, max_tokens)
    }

    fn generate_chat_multi(
        &self,
        messages: &[super::ChatMessage],
        max_tokens: u32,
    ) -> Result<String> {
        let gen_model = self
            .generation_model
            .as_ref()
            .context("generation model not loaded")?;

        let chat_messages: Vec<LlamaChatMessage> = messages.iter()
            .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone())
                .map_err(|e| anyhow::anyhow!("chat message error: {e}")))
            .collect::<Result<Vec<_>>>()?;

        let template = gen_model
            .chat_template(None)
            .map_err(|e| anyhow::anyhow!("no chat template in model: {e}"))?;

        let formatted = gen_model
            .apply_chat_template(&template, &chat_messages, true)
            .map_err(|e| anyhow::anyhow!("chat template failed: {e}"))?;

        self.generate_with_grammar(&formatted, max_tokens)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embed_model = self
            .embedding_model
            .as_ref()
            .context("embedding model not loaded")?;

        let tokens = embed_model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let n_tokens = tokens.len();
        let _n_embd = embed_model.n_embd() as usize;

        // Try safe API first (works with ROCm, crashes with Vulkan on BERT models).
        // Creates a fresh context per call since BERT contexts are lightweight.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_tokens as u32 + 16))
            .with_n_batch(n_tokens as u32)
            .with_embeddings(true)
            .with_flash_attention_policy(0); // disabled for BERT

        // SAFETY: model lives in self and outlives this call
        let model_ref: &'static LlamaModel = unsafe { &*(embed_model as *const LlamaModel) };
        let mut ctx = model_ref
            .new_context(&self._init, ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create embedding context: {e}"))?;

        let mut batch = LlamaBatch::new(n_tokens, 1);
        batch
            .add_sequence(&tokens, 0, true)
            .map_err(|e| anyhow::anyhow!("batch add failed: {e}"))?;

        ctx.encode(&mut batch)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?;

        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("embeddings extraction failed: {e}"))?;

        let result = embedding.to_vec();

        // L2 normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            Ok(result.iter().map(|x| x / norm).collect())
        } else {
            Ok(result)
        }
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}
