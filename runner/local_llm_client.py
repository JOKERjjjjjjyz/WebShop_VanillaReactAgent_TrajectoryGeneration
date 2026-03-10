import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class LocalHuggingFaceClient:
    """
    A client for running locally deployed HuggingFace models (e.g., Qwen/Qwen1.5-0.5B-Chat).
    It uses the Transformers library and assumes you have a GPU available.
    """
    def __init__(self, model_name_or_path: str = "Qwen/Qwen1.5-0.5B-Chat"):
        print(f"Loading local model {model_name_or_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )
        
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto", # Automatically assigns model to GPU if available
            trust_remote_code=True,
            torch_dtype=torch.float16 # Use half-precision for speed/memory
        )
        self.model.eval()
        print(f"Model loaded on {self.device}!")

    def chat(self, history: List[Dict[str, str]]) -> str:
        """
        history format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        # Qwen and similar chat models usually support apply_chat_template
        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate output
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,   # High temperature for diverse plan sampling
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Slicing the output to only return the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
