"""
This predictor is a cog wrapper around the VILA Vision-Language Model. It loads the model into memory and runs predictions on it.
The predict method is based on https://github.com/Efficient-Large-Model/VILA/blob/main/llava/eval/run_llava.py
"""

import os
import subprocess

from cog import BasePredictor, Input, Path
import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from transformers import logging

# Model configuration and download links
MODEL_NAME = "vila-13b" # Choose from ["vila-2.7b", "vila-7b", "vila-13b"]
assert MODEL_NAME in ["vila-2.7b", "vila-7b", "vila-13b"]

MODEL_URL_MAP = {
    "vila-2.7b": "https://weights.replicate.delivery/default/vila-2.7b/vila-2.7b.tar",
    "vila-7b"  : "https://weights.replicate.delivery/default/vila-7b/vila-7b.tar",
    "vila-13b" : "https://weights.replicate.delivery/default/vila-13b/vila-13b.tar",
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        logging.set_verbosity_error() # Hide the transformers warnings

        model_path = MODEL_NAME # Will contain checkpoint files, and config files, etc.

        print("Loading the model...")
        if not os.path.exists(model_path):
            checkpoint_url = MODEL_URL_MAP[MODEL_NAME]
            subprocess.check_output(["pget", "-x", checkpoint_url, model_path], close_fds=False)

        disable_torch_init() # For faster model loading
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, model_name
        )
        print(f"Model Loaded: {model_name}")

    def predict(
        self,
        image: Path = Input(
            description="Image 1 to discuss",
        ),        
        prompt: str = Input(
            description="Query to ask the model about the image",
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            default=1,
            ge=0,
            le=1,
        ),
        temperature: float = Input(
            description="When decoding text, higher values make the model more creative",
            default=0.2,
            ge=0,
        ),
        num_beams: int = Input(
            description="Number of beams to use when decoding text; higher values are slower but more accurate",
            default=1,
            ge=1,
            le=5,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=512,
            ge=1,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        
        # 1 - Prepare the text input
        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if self.model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = (
            tokenizer_image_token(
                prompt, 
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        
        # 2 - Prepare the image input
        images = [Image.open(str(image)).convert("RGB")]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        # 3 - Prepare the stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, 
            self.tokenizer, 
            input_ids
        )
        
        # 4 - Generate the text
        output_ids = self.model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0] # Decode output ids and remove the input text
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)] # Remove the stopping string
            
        return outputs
