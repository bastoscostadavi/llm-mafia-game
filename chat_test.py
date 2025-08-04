#!/usr/bin/env python3
"""
Interactive chat with Llama 3.1 8B for testing
"""
import sys
sys.path.append('.')

from src.agents.llm_interface import LlamaCppInterface

def main():
    print("Loading Llama 3.1 8B...")
    llm = LlamaCppInterface('models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    print("Model loaded! Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() == 'quit':
                break
            if prompt.strip() == '':
                continue
                
            response = llm.generate(prompt, max_tokens=200)
            print(f"Llama: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()