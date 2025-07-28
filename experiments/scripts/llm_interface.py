# src/llm_interface.py
from llama_cpp import Llama
import os

class LlamaCppInterface:
    def __init__(self, model_path: str):
        """Initialize llama.cpp model"""
        print(f"Loading model from: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Use all GPU layers on M4
            n_ctx=2048,
            verbose=False
        )
        print("✅ Model loaded!")
    
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate response"""
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            echo=False
        )
        return response['choices'][0]['text'].strip()

# Test function
if __name__ == "__main__":
    print("Testing LlamaCppInterface...")
    # Update this path to your model
    model_path = "models/mistral.gguf"
    
    if os.path.exists(model_path):
        llm = LlamaCppInterface(model_path)
        response = llm.generate("Hello! Count to 5:")
        print(f"Response: {response}")
    else:
        print(f"❌ Model not found at {model_path}")