# src/llm_utils.py - LLM Interface Utilities
import os
import re
import hashlib

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

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
    def __init__(self, client, model, temperature=0.7, reasoning_effort=None):
        self.client = client
        self.model = model
        self.temperature = temperature
        # Force minimal reasoning effort for gpt-5-mini if not specified
        if model == 'gpt-5-mini' and reasoning_effort is None:
            self.reasoning_effort = 'minimal'
        else:
            self.reasoning_effort = reasoning_effort
        self.display_name = model
    
    def generate(self, prompt, max_tokens=50):
        if self.model.startswith('gpt-5'):
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens
            }
            if self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort
            response = self.client.chat.completions.create(**params)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
        
        choice = response.choices[0]
        
        # Handle empty content - raise exceptions to stop the game
        if choice.message.content is None or choice.message.content.strip() == "":
            if choice.finish_reason == "content_filter":
                raise RuntimeError("Content filtered by AI safety system - game cannot continue")
            elif choice.finish_reason == "length":
                raise RuntimeError("Response truncated due to length limit - game cannot continue")
            else:
                raise RuntimeError("AI model returned empty response - game cannot continue")
        
        return choice.message.content.strip()

class Google:
    """Google Gemini API wrapper"""
    def __init__(self, model, temperature=0.7):
        self.model_name = model
        self.temperature = temperature
        self.display_name = model
        # Configure the API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate(self, prompt, max_tokens=50):
        try:
            # Create generation config with adjusted safety settings
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=self.temperature,
            )
            
            # Configure safety settings to be maximally permissive for research purposes
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check if response was blocked by safety filters
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.finish_reason == 2:  # SAFETY
                    # Extract safety ratings for debugging
                    safety_info = []
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            if hasattr(rating, 'probability') and rating.probability in ['HIGH', 'MEDIUM']:
                                category = getattr(rating, 'category', 'UNKNOWN')
                                prob = getattr(rating, 'probability', 'UNKNOWN')
                                safety_info.append(f"{category}: {prob}")
                    
                    if safety_info:
                        raise RuntimeError(f"Content blocked by Gemini safety filters: {', '.join(safety_info)} - game cannot continue")
                    else:
                        # Try even more permissive settings
                        raise RuntimeError("Content blocked by Gemini safety filters (no specific categories detected). Try adjusting the game prompts to be less adversarial - game cannot continue")
                elif candidate.finish_reason == 3:  # RECITATION
                    raise RuntimeError("Content blocked by Gemini recitation filter - game cannot continue")
            
            if response.text:
                return response.text.strip()
            else:
                raise RuntimeError("Google Gemini returned no response - game cannot continue")
                
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e} - game cannot continue")

class XAI:
    """xAI API wrapper (OpenAI compatible)"""
    def __init__(self, client, model, temperature=0.7):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.display_name = model
    
    def generate(self, prompt, max_tokens=50):
        # Grok-4 is a reasoning model, doesn't support temperature, stop, etc.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
            # Note: Grok-4 doesn't support temperature, presencePenalty, frequencyPenalty, stop
        )
        
        choice = response.choices[0]
        
        # Handle empty content - raise exceptions to stop the game
        if choice.message.content is None or choice.message.content.strip() == "":
            if choice.finish_reason == "content_filter":
                raise RuntimeError("Content filtered by AI safety system - game cannot continue")
            elif choice.finish_reason == "length":
                raise RuntimeError("Response truncated due to length limit - game cannot continue")
            else:
                raise RuntimeError("AI model returned empty response - game cannot continue")
        
        return choice.message.content.strip()

class Anthropic:
    """Anthropic API wrapper with prompt caching support"""
    def __init__(self, client, model, temperature=0.7, use_cache=False):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.use_cache = use_cache
        self.display_name = model
    
    def generate(self, prompt, max_tokens=50):
        # For v4.0 prompts, enable caching on the static game rules section
        if self.use_cache and "#GAME PLAYERS AND COMPOSITION" in prompt:
            # Find the cache boundary - everything before this is static and cacheable
            cache_boundary = prompt.find("#GAME PLAYERS AND COMPOSITION")
            if cache_boundary > 0:
                cacheable_part = prompt[:cache_boundary].strip()
                dynamic_part = prompt[cache_boundary:].strip()
                
                # Check if cacheable part meets minimum token requirement (1024+ for Sonnet)
                # Rough estimate: ~4 chars per token, so need ~4096+ characters
                if len(cacheable_part) >= 4000:
                    messages = [
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": cacheable_part,
                                    "cache_control": {"type": "ephemeral"}
                                },
                                {
                                    "type": "text", 
                                    "text": "\n\n" + dynamic_part
                                }
                            ]
                        }
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
            else:
                # Fallback to regular message if no cache boundary found
                messages = [{"role": "user", "content": prompt}]
        else:
            # Regular message without caching
            messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=messages
        )
        return response.content[0].text.strip()

class Local:
    """Local llama-cpp model wrapper with prompt caching"""
    def __init__(self, model, temperature=0.3):
        self.model = model
        self.model_path = getattr(model, 'model_path', 'unknown')
        self.temperature = temperature
        self.is_gpt_oss = 'gpt-oss' in self.model_path.lower()
        self.prompt_cache = {}  # Cache for processed prompt prefixes
        # Extract model name from path for display
        self.display_name = os.path.basename(self.model_path)
    
    def generate(self, prompt, max_tokens=50):
        """Generate response with prompt caching optimization"""
        
        # Check if we can use prompt caching for v4.0 prompts
        cache_key = None
        if "#GAME PLAYERS AND COMPOSITION" in prompt:
            cache_boundary = prompt.find("#GAME PLAYERS AND COMPOSITION")
            if cache_boundary > 0:
                cacheable_part = prompt[:cache_boundary].strip()
                dynamic_part = prompt[cache_boundary:].strip()
                
                # Create cache key from cacheable part
                cache_key = hashlib.md5(cacheable_part.encode()).hexdigest()
                
                # Check if we have this cached
                if cache_key in self.prompt_cache:
                    # Process only the dynamic part with cached context
                    return self._generate_with_cache(cacheable_part, dynamic_part, max_tokens, cache_key)
                else:
                    pass  # Cache miss, continue to full generation
        
        # For cache miss, we need to set up caching properly
        if cache_key:
            return self._generate_and_cache(cacheable_part, dynamic_part, max_tokens, cache_key)
        
        # No caching, just generate normally
        if self.is_gpt_oss:
            return self._generate_gpt_oss(prompt, max_tokens=10000)
        else:
            return self._generate_standard(prompt, max_tokens)
    
    def _generate_and_cache(self, cacheable_part, dynamic_part, max_tokens, cache_key):
        """Generate response while setting up KV cache for future use - FIXED APPROACH"""
        try:
            # For now, just generate normally and set up caching for next time
            # This avoids breaking the current response
            full_prompt = cacheable_part + "\n\n" + dynamic_part
            
            if self.is_gpt_oss:
                result = self._generate_gpt_oss(full_prompt, max_tokens=10000)
            else:
                result = self._generate_standard(full_prompt, max_tokens)
            
            # After successful generation, set up cache for next time by processing just the cacheable part
            try:
                # Process cacheable part with minimal continuation to get KV state
                cache_setup_prompt = cacheable_part + "\n\n#PLAYER CONTEXT:\nContinue:"
                
                if self.is_gpt_oss:
                    temp_response = self._generate_gpt_oss(cache_setup_prompt, max_tokens=1)
                else:
                    temp_response = self._generate_standard(cache_setup_prompt, max_tokens=1)
                
                # Save the KV state after processing cacheable part
                cached_state = self.model.save_state()
                
                self.prompt_cache[cache_key] = {
                    "state": cached_state,
                    "cached": True
                }
                
            except Exception as cache_error:
                # Mark as processed even if caching failed
                self.prompt_cache[cache_key] = {"processed": True}
            
            return result
            
        except Exception as e:
            # Fallback to full generation without caching
            full_prompt = cacheable_part + "\n\n" + dynamic_part
            if self.is_gpt_oss:
                return self._generate_gpt_oss(full_prompt, max_tokens=10000)
            else:
                return self._generate_standard(full_prompt, max_tokens)
    
    def _generate_standard(self, prompt, max_tokens):
        """Standard generation for non-GPT-OSS models"""
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
        
        # Fallback to generator approach
        response_generator = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            echo=False
        )
        
        full_text = ""
        error_tokens = {'id', 'object', 'created', 'model', 'choices', 'usage'}
        
        for token_data in response_generator:
            if isinstance(token_data, dict) and 'choices' in token_data and len(token_data['choices']) > 0:
                full_text += token_data['choices'][0]['text']
            elif isinstance(token_data, str) and token_data not in error_tokens:
                full_text += token_data
        
        return full_text.strip()
    
    def _generate_with_cache(self, cacheable_part, dynamic_part, max_tokens, cache_key):
        """Generate response using cached KV state - TRUE CACHING"""
        try:
            # Load the cached KV state
            cached_state = self.prompt_cache[cache_key]['state']
            self.model.load_state(cached_state)
            
            # Generate response with only the dynamic part
            # The model already has the static context in KV cache
            response = self._generate_standard(dynamic_part, max_tokens)
            return response
            
        except Exception as e:
            # Fallback to full generation
            full_prompt = cacheable_part + "\n\n" + dynamic_part
            return self._generate_standard(full_prompt, max_tokens)
    
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
            raise RuntimeError(f"GPT-OSS generation failed: {e} - game cannot continue")

class Human:
    """Human input interface - displays prompts and captures user responses"""
    def __init__(self, player_name):
        self.player_name = player_name
        self.display_name = f"Human Player ({player_name})"

    def generate(self, prompt, max_tokens=50):
        """Display clean prompt to human and capture their response"""
        print(f"\n{prompt}")

        try:
            response = input(f"\n> ").strip()
            if not response:
                return "I have no response"
            return response
        except KeyboardInterrupt:
            print(f"\n{self.player_name} left the game.")
            raise RuntimeError(f"Human player {self.player_name} interrupted the game - stopping")
        except EOFError:
            raise RuntimeError(f"Human player {self.player_name} input ended unexpectedly - stopping")

def get_model_display_name(llm_wrapper):
    """Get human-readable model name from LLM wrapper"""
    if hasattr(llm_wrapper, 'model'):
        if isinstance(llm_wrapper.model, str):
            # OpenAI/Anthropic wrappers store model name as string
            return llm_wrapper.model
        # Local wrapper has Llama object as model
    if hasattr(llm_wrapper, 'model_path'):
        # Extract model name from path
        return os.path.basename(llm_wrapper.model_path)
    return "Unknown Model"

def create_agent_interface(llm_config):
    """Create agent interface - supports AI models (local, openai, anthropic, etc.) and human players"""
    llm_type = llm_config.get('type', 'local')
    
    if llm_type == 'local':
        # Use shared model to avoid conflicts
        model_filename = llm_config.get('model', 'mistral.gguf')
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'models', model_filename)
        
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
            return create_agent_interface({'type': 'local'})
        
        client = openai.OpenAI(api_key=llm_config.get('api_key') or os.getenv('OPENAI_API_KEY'))
        model = llm_config.get('model', 'gpt-3.5-turbo')
        temperature = llm_config.get('temperature', 0.7)
        reasoning_effort = llm_config.get('reasoning_effort')
        return OpenAI(client, model, temperature, reasoning_effort)
    
    elif llm_type == 'xai':
        if openai is None:
            print("ERROR: OpenAI package not installed (needed for xAI). Using local model instead.")
            return create_agent_interface({'type': 'local'})
        
        # xAI uses OpenAI SDK with different base URL
        client = openai.OpenAI(
            api_key=llm_config.get('api_key') or os.getenv('XAI_API_KEY'),
            base_url="https://api.x.ai/v1"
        )
        model = llm_config.get('model', 'grok-4')
        temperature = llm_config.get('temperature', 0.7)
        return XAI(client, model, temperature)
    
    elif llm_type == 'deepseek':
        if openai is None:
            print("ERROR: OpenAI package not installed (needed for DeepSeek). Using local model instead.")
            return create_agent_interface({'type': 'local'})
        
        # DeepSeek uses OpenAI SDK with different base URL
        client = openai.OpenAI(
            api_key=llm_config.get('api_key') or os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        model = llm_config.get('model', 'deepseek-v3.1')
        temperature = llm_config.get('temperature', 0.7)
        return OpenAI(client, model, temperature)
    
    elif llm_type == 'anthropic':
        if anthropic is None:
            print("ERROR: Anthropic package not installed. Using local model instead.")
            return create_agent_interface({'type': 'local'})
        
        client = anthropic.Anthropic(api_key=llm_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'))
        model = llm_config.get('model', 'claude-3-haiku-20240307')
        temperature = llm_config.get('temperature', 0.7)
        use_cache = llm_config.get('use_cache', False)
        return Anthropic(client, model, temperature, use_cache)
    
    elif llm_type == 'google':
        if genai is None:
            print("ERROR: Google GenerativeAI package not installed. Using local model instead.")
            return create_agent_interface({'type': 'local'})
        
        model = llm_config.get('model', 'gemini-2.5-flash-lite')
        temperature = llm_config.get('temperature', 0.7)
        return Google(model, temperature)
    
    elif llm_type == 'human':
        player_name = llm_config.get('player_name', 'Human Player')
        return Human(player_name)
    
    else:
        # Default to local
        return create_agent_interface({'type': 'local'})