"""
Utility to load environment variables from .env file
"""

import os
from pathlib import Path

def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = Path('.env')
    
    if env_path.exists():
        print("Loading environment variables from .env file...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✓ Environment variables loaded")
    else:
        print("No .env file found. Using system environment variables.")

if __name__ == "__main__":
    load_env()
    
    # Check which API keys are available
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "COHERE_API_KEY": "Cohere", 
        "ANTHROPIC_API_KEY": "Anthropic",
        "HUGGINGFACE_API_TOKEN": "HuggingFace"
    }
    
    print("\nAPI Key Status:")
    for key, name in api_keys.items():
        if os.getenv(key):
            print(f"✓ {name} API key is set")
        else:
            print(f"✗ {name} API key is not set")