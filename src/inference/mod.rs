pub mod gpu_check;
pub mod llama;

use anyhow::Result;

/// A role/content pair for multi-turn chat.
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

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

    /// Generate a chat completion with arbitrary multi-turn messages.
    /// Messages should include system, user, assistant turns in order.
    /// The model will generate the next assistant turn.
    fn generate_chat_multi(
        &self,
        messages: &[ChatMessage],
        max_tokens: u32,
    ) -> Result<String> {
        // Default: fall back to generate_chat with last user message
        let system = messages.iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        let user = messages.iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        self.generate_chat(system, user, max_tokens)
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
