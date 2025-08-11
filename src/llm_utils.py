# src/llm_utils.py - LLM Interface Utilities
import os
import re

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

class OpenAI:
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

class Anthropic:
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

class Local:
    """Local llama-cpp model wrapper"""
    def __init__(self, model, temperature=0.3):
        self.model = model
        self.model_path = getattr(model, 'model_path', 'unknown')
        self.temperature = temperature
        self.is_gpt_oss = 'gpt-oss' in self.model_path.lower()
    
    def generate(self, prompt, max_tokens=50):
        """Generate response handling llama-cpp generator properly"""
        
        # Special handling for GPT-OSS reasoning models
        if self.is_gpt_oss:
            # Let GPT-OSS think with very high token limit
            return self._generate_gpt_oss(prompt, max_tokens=4000)
        
        # Standard handling for other models
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
    
    def _generate_gpt_oss(self, prompt, max_tokens=50):
        """Simple GPT-OSS handling - extract final channel if present, otherwise return full response"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            result = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            if 'choices' in result and len(result['choices']) > 0:
                response = result['choices'][0]['message']['content'].strip()
                
                # Simple: if there's a final channel, use that
                if 'final<|message|>' in response:
                    return response.split('final<|message|>')[-1]
                
                # Otherwise return full response
                return response
                    
        except Exception as e:
            print(f"GPT-OSS generation failed: {e}")
        
        return "I understand the situation."

class Human:
    """Human input interface - displays prompts and captures user responses"""
    def __init__(self, player_name):
        self.player_name = player_name
    
    def generate(self, prompt, max_tokens=50):
        """Display prompt to human and capture their response"""
        print(f"\n{'='*60}")
        print(f"PROMPT FOR {self.player_name.upper()}:")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}")
        
        try:
            response = input(f"\n{self.player_name}, your response: ").strip()
            if not response:
                return "No response"
            return response
        except KeyboardInterrupt:
            print(f"\n{self.player_name} left the game.")
            return "No response"
        except EOFError:
            return "No response"

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
        return Local(_model_cache[model_path], temperature)
    
    elif llm_type == 'openai':
        if openai is None:
            print("ERROR: OpenAI package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
        
        client = openai.OpenAI(api_key=llm_config.get('api_key') or os.getenv('OPENAI_API_KEY'))
        model = llm_config.get('model', 'gpt-3.5-turbo')
        temperature = llm_config.get('temperature', 0.7)
        return OpenAI(client, model, temperature)
    
    elif llm_type == 'anthropic':
        if anthropic is None:
            print("ERROR: Anthropic package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
        
        client = anthropic.Anthropic(api_key=llm_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'))
        model = llm_config.get('model', 'claude-3-haiku-20240307')
        temperature = llm_config.get('temperature', 0.7)
        return Anthropic(client, model, temperature)
    
    elif llm_type == 'human':
        player_name = llm_config.get('player_name', 'Human Player')
        return Human(player_name)
    
    else:
        # Default to local
        return create_llm({'type': 'local'})