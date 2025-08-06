# src/llm_utils.py - LLM Interface Utilities
import os

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from dotenv import load_dotenv
    # Load .env file if it exists
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading
    pass

# Simple model cache to avoid conflicts
_model_cache = {}

class SimpleOpenAI:
    """OpenAI API wrapper"""
    def __init__(self, client, model, temperature=0.7):
        self.client = client
        self.model = model
        self.temperature = temperature
    
    def generate(self, prompt, max_tokens=50):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

class SimpleAnthropic:
    """Anthropic API wrapper"""
    def __init__(self, client, model, temperature=0.7):
        self.client = client
        self.model = model
        self.temperature = temperature
    
    def generate(self, prompt, max_tokens=50):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

class SimpleLocal:
    """Local llama-cpp model wrapper"""
    def __init__(self, model, temperature=0.3):
        self.model = model
        self.model_path = getattr(model, 'model_path', 'unknown')
        self.temperature = temperature
    
    def generate(self, prompt, max_tokens=50):
        """Generate response handling llama-cpp generator properly"""
        # Try the completion method instead of __call__
        try:
            result = self.model.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=False
            )
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['text'].strip()
        except Exception as e:
            print(f"DEBUG: create_completion failed: {e}")
        
        # Fallback to original generator approach but skip the problematic response
        response_generator = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        # Collect all text from the generator, but ignore obvious error tokens
        full_text = ""
        error_tokens = {'id', 'object', 'created', 'model', 'choices', 'usage'}
        
        for token_data in response_generator:
            if isinstance(token_data, dict) and 'choices' in token_data and len(token_data['choices']) > 0:
                full_text += token_data['choices'][0]['text']
            elif isinstance(token_data, str) and token_data not in error_tokens:
                full_text += token_data
        
        return full_text.strip()

def create_llm(llm_config):
    """Simple LLM creator - supports local, openai, anthropic"""
    llm_type = llm_config.get('type', 'local')
    
    if llm_type == 'local':
        # Use shared model to avoid conflicts
        model_path = llm_config.get('model_path', 'models/mistral.gguf')
        if model_path not in _model_cache:
            from llama_cpp import Llama
            print(f"Loading model from: {model_path}")
            model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Use all GPU layers
                n_ctx=llm_config.get('n_ctx', 2048),  # Configurable context window
                verbose=False
            )
            print("Model loaded!")
            _model_cache[model_path] = model
        temperature = llm_config.get('temperature', 0.3)
        return SimpleLocal(_model_cache[model_path], temperature)
    
    elif llm_type == 'openai':
        if openai is None:
            print("ERROR: OpenAI package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
        
        client = openai.OpenAI(api_key=llm_config.get('api_key') or os.getenv('OPENAI_API_KEY'))
        model = llm_config.get('model', 'gpt-3.5-turbo')
        temperature = llm_config.get('temperature', 0.7)
        return SimpleOpenAI(client, model, temperature)
    
    elif llm_type == 'anthropic':
        if anthropic is None:
            print("ERROR: Anthropic package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
        
        client = anthropic.Anthropic(api_key=llm_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'))
        model = llm_config.get('model', 'claude-3-haiku-20240307')
        temperature = llm_config.get('temperature', 0.7)
        return SimpleAnthropic(client, model, temperature)
    
    else:
        # Default to local
        return create_llm({'type': 'local'})