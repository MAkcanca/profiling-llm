import os
import json
import time
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, DefaultDict
from collections import defaultdict

from baml_client import b
from baml_client.types import ProfileResult
from baml_py import ClientRegistry
from dotenv import load_dotenv

load_dotenv()

# Model configurations to test
MODEL_CONFIGS = [
    {
        "name": "GPT-4.5-Preview",
        "provider": "openai",
        "options": {
            "model": "gpt-4.5-preview-2025-02-27",
            "api_key": os.environ.get('OPENAI_API_KEY'),
        }
    },
    {
        "name": "o3-mini",
        "provider": "openai",
        "options": {
            "model": "o3-mini",
            "api_key": os.environ.get('OPENAI_API_KEY')
        }
    },
    {
        "name": "o3-mini-high",
        "provider": "openai",
        "options": {
            "model": "o3-mini",
            "api_key": os.environ.get('OPENAI_API_KEY'),
            "reasoning_effort": "high"
        }
    },
    {
        "name": "GPT-4o",
        "provider": "openai",
        "options": {
            "model": "gpt-4o",
            "api_key": os.environ.get('OPENAI_API_KEY')
        }
    },
    {
        "name": "GPT-4o-mini",
        "provider": "openai",
        "options": {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get('OPENAI_API_KEY')
        }
    },
    {
        "name": "Claude-3.7-Sonnet",
        "provider": "anthropic",
        "options": {
            "model": "claude-3-7-sonnet-20250219",
            "api_key": os.environ.get('ANTHROPIC_API_KEY')
        }
    },
    {
        "name": "Claude-3.7-Sonnet-Thinking",
        "provider": "anthropic",
        "options": {
            "model": "claude-3-7-sonnet-20250219",
            "api_key": os.environ.get('ANTHROPIC_API_KEY'),
            "max_tokens": 16000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 8192
            }
        }
    },
    {
        "name": "Llama-3.3-70B-Instruct",
        "provider": "openai-generic",
        "options": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.3-70b-instruct",
            "api_key": os.environ.get('OPENROUTER_API_KEY')
        }
    },
    {
        "name": "DeepSeek-R1",
        "provider": "openai-generic",
        "options": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "deepseek/deepseek-r1",
            "api_key": os.environ.get('OPENROUTER_API_KEY')
        }
    },
    {
        "name": "Gemini-2.0-Flash",
        "provider": "google-ai",
        "options": {
            "model": "gemini-2.0-flash",
            "api_key": os.environ.get('GEMINI_API_KEY')
        }
    },
    {
        "name": "Gemini-2.0-Flash-Thinking-Exp0121",
        "provider": "google-ai",
        "options": {
            "model": "gemini-2.0-flash-thinking-exp-01-21",
            "api_key": os.environ.get('GEMINI_API_KEY')
        }
    },
    {
        "name": "Gemma-3",
        "provider": "openai-generic",
        "options": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "google/gemma-3-27b-it",
            "api_key": os.environ.get('OPENROUTER_API_KEY')
        }
    }
]

class ProfileGenerator:
    """Class to generate criminal profiles using different LLM models."""
    
    def __init__(self, output_dir="generated_profiles"):
        """Initialize the profile generator.
        
        Args:
            output_dir: Directory to store generated profiles
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create timestamped run directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Store generation metadata
        self.generation_metadata = []
    
    def load_test_case(self, test_case_path: str) -> str:
        """Load a test case from file.
        
        Args:
            test_case_path: Path to the test case file
            
        Returns:
            The test case content as a string
        """
        with open(test_case_path, "r") as f:
            return f.read()
    
    async def _process_single_extraction(self, test_case_name: str, case_data: str, 
                                        model_config: Dict, case_dir: Path) -> None:
        """Process a single extraction task for a specific model and test case."""
        print(f"Generating profile with {model_config['name']} for {test_case_name}...")
        
        # Set up client registry
        cr = ClientRegistry()
        cr.add_llm_client(
            name=model_config['name'],
            provider=model_config['provider'],
            options=model_config['options']
        )
        cr.set_primary(model_config['name'])
        
        start_time = time.time()
        
        # Retry conf
        max_retries = 2  # Will try up to 3 times total (initial + 2 retries)
        retry_count = 0
        base_delay = 30.0  # Secs
        
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 5)
                    print(f"Retry {retry_count}/{max_retries} for {model_config['name']} on {test_case_name} after {delay:.2f}s delay...")
                    await asyncio.sleep(delay)
                
                # Generate the profile asynchronously
                profile_result: ProfileResult = await b.ExtractProfile(case_data, {"client_registry": cr})
                end_time = time.time()
                processing_time = end_time - start_time
                
                result_file = case_dir / f"{model_config['name']}_result.json"
                with open(result_file, "w") as f:
                    json.dump(profile_result.__dict__, f, default=lambda o: o.__dict__, indent=2)
                
                # Store metadata
                retry_info = {"retries": retry_count} if retry_count > 0 else {}
                self.generation_metadata.append({
                    "test_case": test_case_name,
                    "model": model_config['name'],
                    "processing_time": processing_time,
                    "result_file": str(result_file),
                    "timestamp": datetime.now().isoformat(),
                    **retry_info
                })
                
                # Successful completion
                if retry_count > 0:
                    print(f"Successfully completed {model_config['name']} on {test_case_name} after {retry_count} retries")
                return
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"Error with {model_config['name']} on {test_case_name} (attempt {retry_count}/{max_retries + 1}): {e}")
                else:
                    # Final failure after all retries - sad case
                    print(f"Final error with {model_config['name']} on {test_case_name} after {max_retries} retries: {e}")
                    self.generation_metadata.append({
                        "test_case": test_case_name,
                        "model": model_config['name'],
                        "error": str(e),
                        "retries": retry_count - 1,
                        "timestamp": datetime.now().isoformat(),
                    })
    
    async def _process_test_case(self, test_case_path: str, models: List[Dict], 
                               max_concurrent_per_provider: Dict[str, int]) -> None:
        """Process a single test case with all models."""
        test_case_name = Path(test_case_path).stem
        case_data = self.load_test_case(test_case_path)
        
        print(f"Processing test case: {test_case_name}")
        
        # Create dir
        case_dir = self.run_dir / test_case_name
        case_dir.mkdir(exist_ok=True)
        
        # Group models by provider
        models_by_provider = defaultdict(list)
        for model_config in models:
            provider = model_config['provider']
            models_by_provider[provider].append(model_config)
        
        # Process each provider's models with controlled concurrency
        for provider, provider_models in models_by_provider.items():
            print(f"  Processing {len(provider_models)} models for provider: {provider}")
            
            # Process models in batches
            tasks = []
            for model_config in provider_models:
                # Create task for this model
                task = self._process_single_extraction(
                    test_case_name, case_data, model_config, case_dir
                )
                tasks.append(task)
                
                # If we've reached max concurrency for this provider, wait for this batch to complete
                provider_max_concurrent = max_concurrent_per_provider.get(provider, 2)
                if len(tasks) >= provider_max_concurrent:
                    await asyncio.gather(*tasks)
                    tasks = []
            
            # Wait for any remaining tasks to complete
            if tasks:
                await asyncio.gather(*tasks)
    
    async def generate_profiles(self, test_cases: List[str], models: Optional[List[Dict]] = None, 
                              max_concurrent_per_provider: Dict[str, int] = None) -> str:
        """Generate profiles for all test cases using specified models.
        
        Args:
            test_cases: List of paths to test case files
            models: List of model configurations to use (defaults to MODEL_CONFIGS)
            max_concurrent_per_provider: Dict mapping provider names to max concurrent extractions
                Default is 2 concurrent extractions per provider
            
        Returns:
            Path to the directory containing generated profiles
        """
        if models is None:
            models = MODEL_CONFIGS
            
        # Default concurrency settings if not specified
        if max_concurrent_per_provider is None:
            max_concurrent_per_provider = defaultdict(lambda: 2)
        
        for test_case_path in test_cases:
            await self._process_test_case(test_case_path, models, max_concurrent_per_provider)
        
        # Save the generation metadata
        metadata_file = self.run_dir / "generation_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({
                "timestamp": self.run_timestamp,
                "metadata": self.generation_metadata
            }, f, indent=2)
        
        return str(self.run_dir)
    
    async def generate_profiles_parallel_cases(self, test_cases: List[str], models: Optional[List[Dict]] = None,
                                            max_concurrent_per_provider: Dict[str, int] = None,
                                            max_concurrent_cases: int = 2) -> str:
        """Generate profiles for all test cases in parallel.
        
        Args:
            test_cases: List of paths to test case files
            models: List of model configurations to use (defaults to MODEL_CONFIGS)
            max_concurrent_per_provider: Dict mapping provider names to max concurrent extractions
            max_concurrent_cases: Maximum number of test cases to process in parallel
            
        Returns:
            Path to the directory containing generated profiles
        """
        if models is None:
            models = MODEL_CONFIGS
            
        # Default concurrency settings if not specified
        if max_concurrent_per_provider is None:
            max_concurrent_per_provider = defaultdict(lambda: 2)
        
        # Process test cases in parallel batches
        for i in range(0, len(test_cases), max_concurrent_cases):
            batch = test_cases[i:i + max_concurrent_cases]
            tasks = [self._process_test_case(tc, models, max_concurrent_per_provider) for tc in batch]
            await asyncio.gather(*tasks)
        
        # Save the generation metadata
        metadata_file = self.run_dir / "generation_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({
                "timestamp": self.run_timestamp,
                "metadata": self.generation_metadata
            }, f, indent=2)
        
        return str(self.run_dir)

async def main_async():
    """Generate profiles for testing asynchronously."""
    # Define test cases
    test_cases = [
        "test-cases/ted-bundy-lake.md",
        "test-cases/mad-bomber.md",
        "test-cases/unabomber.md",
        "test-cases/robert-napper-rachel-nickell.md",
        "test-cases/ed-kemper.md",
        "test-cases/btk-otero.md"
    ]
    
    # Configure concurrency per provider
    provider_concurrency = {
        "openai": 2,        # 2 concurrent OpenAI calls
        "anthropic": 2,      # 2 concurrent Anthropic calls
        "google-ai": 2,      # 2 concurrent Google AI calls
        "openai-generic": 6  # 6 concurrent OpenRouter calls because why not
    }
    
    use_parallel_cases = True
    
    generator = ProfileGenerator()
    
    if use_parallel_cases:
        # Each test case also processes providers in parallel with provider-specific limits
        output_dir = await generator.generate_profiles_parallel_cases(
            test_cases, 
            max_concurrent_per_provider=provider_concurrency,
            max_concurrent_cases=6  # Process up to 6 test cases in parallel
        )
    else:
        # Process test cases sequentially, but parallelize by provider
        output_dir = await generator.generate_profiles(
            test_cases, 
            max_concurrent_per_provider=provider_concurrency
        )
    
    print(f"\nProfiles generated and saved to: {output_dir}")

def main():
    """Generate profiles for testing."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 