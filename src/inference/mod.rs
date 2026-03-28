pub mod llama;

use anyhow::Result;

/// Trait abstracting LLM inference for text generation and embedding.
/// Implementations must be Send + Sync to allow sharing across async tasks.
pub trait InferenceBackend: Send + Sync {
    /// Generate text completion from a prompt.
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String>;

    /// Generate a chat completion with system and user messages.
    /// Uses the model's built-in chat template for proper formatting.
    fn generate_chat(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<String> {
        // Default: concatenate and use raw generate
        let prompt = format!("{system}\n\n{user}");
        self.generate(&prompt, max_tokens)
    }

    /// Produce an embedding vector for a single text input.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Produce embedding vectors for multiple texts.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Return the embedding dimensionality.
    fn embedding_dim(&self) -> usize;
}
