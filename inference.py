"""
Below is the copyright notice from Google.

Please also follow this license when you modify or distribute the code.
"""

"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse

from diffusers import DiffusionPipeline, DDIMScheduler
from transformers import AutoTokenizer
import torch

import os

class MagicTailorInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained_model_name_or_path", type=str,
            default="stabilityai/stable-diffusion-2-1-base",
        )
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--prompt", type=str, required=True)
        parser.add_argument("--output_path", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,   
            torch_dtype=torch.float16,
        )
        self.pipeline.load_lora_weights(self.args.model_path)

        token_embedding_path = os.path.join(self.args.model_path, 'token_embedding.pth')
        token_embedding_state_dict = torch.load(token_embedding_path)
        self.pipeline.text_encoder.get_input_embeddings().weight.data = \
            token_embedding_state_dict['weight'].type(torch.float16)

        self.pipeline.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.args.model_path, 'tokenizer'), 
            use_fast=False
        )

        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.num_inference_steps = 50
        self.guidance_scale = 7.5

        self.pipeline.enable_vae_slicing()
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts):
        images = self.pipeline(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
        if not self.args.output_path:
            self.args.output_path = os.path.join(self.args.model_path, "inference/result.jpg")
        os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)
        images[0].save(self.args.output_path)
        print(f"The genearated image is saved to: {self.args.output_path}")


if __name__ == "__main__":
    inference = MagicTailorInference()
    inference.infer_and_save(
        prompts=[inference.args.prompt]
    )
