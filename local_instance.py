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
            # Local SGLang models (default)
            self.base_url = "http://127.0.0.1:30000/v1"
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
        
        system_prompt = system_prompt + "\n" + """\
You are a world-class expert in hardware design and verification who uses SystemVerilog fluently.
Please THINK HARD on the input.
When you are done, please try to answer the question, preferrably by writing a complete module or function,
including some of the wrapper code you are given, instead of just filling in the blanks.
    """
    
        print("SYSTEM", system_prompt)
        
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
                    {"role": "user", "content": prompt},
                    #{"role": "assistant", "content": "<think>"},
                ],
                max_tokens=32768,
                timeout=timeout,
                temperature=0.7
            )           
            
            print(response)
            
            content = response.choices[0].message.content 
            print(content)
            
            final_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            if "</think>" not in content or not final_content:
                print("COULD NOT FIND THINK IN OUTPUT!, Querying LLM API again...")
                
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
        
            print(final_content)     
            print(files)
            
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
    print(local_instance.prompt("Just write a SystemVerilog module for a 2-to-1 multiplexer. dont only think", files=["answer.txt"], category=9))