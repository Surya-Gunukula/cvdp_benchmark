import os
import logging 
import re
import json 
import sys
from typing import Optional, Any, Dict, List
from src.config_manager import config 

import openai 
import anthropic

#from sglang.utils import print_highlight

import boto3
import json 
from botocore.exceptions import ClientError

# Import ModelHelpers - used for processing responses
try:
    from src.model_helpers import ModelHelpers
except ImportError:
    try:
        # Try alternate import path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.model_helpers import ModelHelpers
    except (ImportError, NameError):
        # Define minimal ModelHelpers class for standalone usage
        class ModelHelpers:
            def create_system_prompt(self, base_context, schema=None, category=None):
                context = base_context if base_context is not None else ""
                if schema is not None:
                    if isinstance(schema, list):
                        context += f"\nProvide the response in one of the following JSON schemas: \n"
                        schemas = []
                        for sch in schema:
                            schemas.append(f"{sch}")
                        context += "\nor\n".join(schemas)
                    else:
                        context += f"\nProvide the response in the following JSON schema: {schema}"
                    context += "\nThe response should be in JSON format, including double-quotes around keys and values, and proper escaping of quotes within values, and escaping of newlines."
                return context
                
            def parse_model_response(self, content, files=None, expected_single_file=False):
                if expected_single_file:
                    return content
                return content
                
            def fix_json_formatting(self, content):
                try:
                    content = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', content)
                    content = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9])(\s*[,}])', r': "\1"\2', content)
                    try:
                        json.loads(content)
                    except json.JSONDecodeError:
                        pass
                except:
                    pass
                return content

logging.basicConfig(level=logging.INFO)

class UnifiedModelInstance:
    """    
    This provides a unified interface for both local models (via SGLang) and Fireworks API models.
    """
    
    def __init__(self, context: Any = "You are a helpful assistant.", key: Optional[str] = None, 
                 model: str = "qwen/qwen2.5-0.5b-instruct"):
        """
        Initialize a model instance.
        
        Args:
            context: The system prompt or context for the model
            key: API key (required for remote models, ignored for local)
            model: The model to use - determines the API endpoint automatically
        """
        self.context = context
        self.model = model
        self.debug = False
        
        # Determine API configuration based on model name
        if model.startswith("accounts/fireworks/"):
            # Fireworks models
            self.base_url = "https://api.fireworks.ai/inference/v1"
            self.api_key = key or config.get("FIREWORKS_API_KEY")
            if self.api_key is None:
                raise ValueError("Unable to connect to Fireworks - No API key provided")
            self.is_local = False
        elif model.startswith("kimi"):
            self.base_url = "https://api.moonshot.ai/v1"
            self.api_key = key or config.get("MOONSHOT_API_KEY")
            if self.api_key is None:
                raise ValueError("Unable to connect to Moonshot - No API key provided")
            self.is_local = False
        else:
            # Local SGLang models (default) - using SFT model on port 8001
            self.base_url = "http://127.0.0.1:8001/v1"
            self.api_key = "None"
            self.is_local = True
        
        self.client = openai.Client(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0
        )
        
        model_type = "Local" if self.is_local else "Remote"
        logging.info(f"Created {model_type} Model using {self.base_url}. Using model: {self.model}")
        
    def set_debug(self, debug: bool = True) -> None:
        """
        Enable or disable debug mode.
        """
        self.debug = debug
        
    def prompt(self, prompt: str, schema: Optional[str] = None, prompt_log: str = "", 
               files: Optional[list] = None, timeout: int = 60, category: Optional[int] = None) -> str:
        """
        Send a prompt to the model and get a response.
        
        Args:
            prompt: The user prompt/query
            schema: Optional JSON schema for structured output
            prompt_log: Path to log the prompt (if not empty)
            files: List of expected output files (if any)
            timeout: Timeout in seconds for the API call (default: 60)
            category: Optional integer indicating the category/problem ID
            
        Returns:
            The model's response as text
        """
        
        
        if hasattr(self, 'client') is False:
            raise ValueError("Model client not initialized")
        
        helper = ModelHelpers()
        system_prompt = helper.create_system_prompt(self.context, schema, category)
        
        system_prompt = """
You are a world-class expert in hardware design and verification who uses SystemVerilog fluently.
Please always THINK HARD and put your thoughts in <think> and </think> tags.
When you are done thinking, please try to answer the question, preferrably by writing a complete module or function,
including some of the wrapper code you are given, instead of just filling in the blanks.
Your response should be:
<think>
thinking here
</think>
your answer here
    """
    
        print("\n" + "="*80)
        print("📋 SYSTEM PROMPT")
        print("="*80)
        print(system_prompt)
        print("="*80 + "\n")
        
        if timeout == 60:
            timeout = config.get("MODEL_TIMEOUT", 60)
            
        # Determine if we're expecting a single file (direct text mode)
        expected_single_file = files and len(files) == 1 and schema is None
        expected_file_name = files[0] if expected_single_file else  None
        
        if prompt_log != "":
            try:
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(system_prompt + "\n\n----------------------------------------\n" + prompt)
                
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log to {prompt_log}: {str(e)}")
                
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + "/think"},
                    # {"role": "assistant", "content": "<think>"},
                ],
                max_tokens=32768,
                timeout=timeout,
                temperature=0.7
            )           
            
            print("\n" + "="*80)
            print("🔌 RAW API RESPONSE")
            print("="*80)
            print(f"Model: {response.model}")
            if hasattr(response, 'usage') and response.usage:
                print(f"Tokens - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
            print("="*80 + "\n")
            
            content = response.choices[0].message.content 
            
            print("\n" + "="*80)
            print("🤖 MODEL OUTPUT (RAW)")
            print("="*80)
            print(content)
            print("="*80 + "\n")
            
            # Check for thinking tags and extract content
            has_opening_tag = "<think>" in content
            has_closing_tag = "</think>" in content
            
            print("\n" + "="*80)
            print("🔍 THINKING TAGS ANALYSIS")
            print("="*80)
            print(f"Has opening tag <think>: {has_opening_tag}")
            print(f"Has closing tag </think>: {has_closing_tag}")
            
            # Try to extract thinking content
            if has_opening_tag and has_closing_tag:
                thinking_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                if thinking_match:
                    thinking_content = thinking_match.group(1).strip()
                    print(f"\n💭 THINKING CONTENT (extracted from tags):")
                    print("-" * 80)
                    print(thinking_content)
                    print("-" * 80)
            elif has_closing_tag and not has_opening_tag:
                # Only closing tag found - extract everything before the closing tag
                closing_pos = content.find("</think>")
                if closing_pos > 0:
                    thinking_content = content[:closing_pos].strip()
                    print(f"\n💭 THINKING CONTENT (extracted - missing opening tag):")
                    print("-" * 80)
                    print(thinking_content)
                    print("-" * 80)
                    print("⚠️  WARNING: Missing opening tag <think>!")
            elif has_opening_tag and not has_closing_tag:
                # Only opening tag found
                opening_pos = content.find("<think>")
                if opening_pos >= 0:
                    thinking_content = content[opening_pos + len("<think>"):].strip()
                    print(f"\n💭 THINKING CONTENT (extracted - missing closing tag):")
                    print("-" * 80)
                    print(thinking_content)
                    print("-" * 80)
                    print("⚠️  WARNING: Missing closing tag </think>!")
            else:
                print("\n⚠️  No thinking tags found in output!")
            print("="*80 + "\n")
            
            final_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            if "</think>" not in content or not final_content:
                print("\n" + "⚠️ " * 40)
                print("⚠️  WARNING: Could not find </think> tags in output!")
                print("⚠️  Querying LLM API again for final response...")
                print("⚠️ " * 40 + "\n")
                
                content += "</think>"
                
                final_prompt = f"{content}\n\nNow, based on the above reasoning, I will produce the final Verilog code in the .sv file."
            
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}, 
                        {"role": "assistant", "content": final_prompt},
                    ],
                    max_tokens=16384,
                    timeout=timeout,
                    temperature=0.7
                )
                
                final_content = final_response.choices[0].message.content
                
                print("\n" + "="*80)
                print("🔄 FINAL RESPONSE (from second API call)")
                print("="*80)
                print(final_content)
                print("="*80 + "\n")
        
            print("\n" + "="*80)
            print("✅ FINAL PROCESSED OUTPUT")
            print("="*80)
            print(final_content)
            print("="*80)
            print(f"\n📁 Expected files: {files}")
            print("="*80 + "\n")
            
            # Process the response using the ModelHelpers
            if expected_single_file:
                pass
            elif schema is not None and final_content.startswith('{') and final_content.endswith('}'):
                final_content = helper.fix_json_formatting(final_content)
            
            
            
            parsed_response, success = helper.parse_model_response(final_content, files, expected_single_file)
            
            # Return outputs, success, rawoutput
            return parsed_response, success, final_content
            
        except Exception as e:
            logging.error(f"Error in prompt: {str(e)}")
            return None
        
        
if __name__ == "__main__":
    # Example usage for local model
    local_instance = UnifiedModelInstance(model="qwen/qwen2.5-14b-instruct")
    print(local_instance.prompt("Write a SystemVerilog module for a 2-to-1 multiplexer.", files=["answer.txt"], category=9))