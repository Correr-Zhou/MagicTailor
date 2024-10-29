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
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import numpy as np

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import ptp_utils
from ptp_utils import AttentionStore
# from diffusers.models.cross_attention import CrossAttention
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm

import cv2

from diffusers.loaders import LoraLoaderMixin
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils.torch_utils import is_compiled_module

from typing import Any, List, Optional, Union
import math

from tools.mask_generation import check_mask_existence, generate_masks, get_gdino_and_sam_model, get_class_names


check_min_version("0.12.0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")

    # data
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/magictailor",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # pipeline
    parser.add_argument(
        "--phase1_train_steps",
        type=int,
        default=200,
        help="Number of trainig steps for the first phase (warm-up).",
    )
    parser.add_argument(
        "--phase2_train_steps",
        type=int,
        default=300,
        help="Number of trainig steps for the second phase (DS-Bal).",
    )
    parser.add_argument(
        "--phase1_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the first phase (warm-up).",
    )
    parser.add_argument(
        "--phase2_learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the second phase (DS-Bal).",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument("--lora_rank", type=int, default=32)

    # cross-attention loss
    parser.add_argument("--lambda_attention", type=float, default=1e-2)

    # DM-Deg
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=32)

    # DS-Bal
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--lambda_preservation", type=float, default=0.2)

    # ckpt
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
         " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )

    # seed
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="A seed for reproducible training."
    )

    # resolution
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=
        ("The resolution for input images"
         " resolution"),
    )

    # model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=
        ("Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
         " float32 precision."),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    # Grounding SAM
    parser.add_argument(
        "--gsam_repo_dir",
        default="Grounded-Segment-Anything",
        type=str,
        help="dir to gsam repo",
    )
    parser.add_argument("--box_threshold",
                        type=float,
                        default=0.25,
                        help="box threshold")
    parser.add_argument("--text_threshold",
                        type=float,
                        default=0.25,
                        help="text threshold")
    parser.add_argument("--nms_threshold",
                        type=float,
                        default=0.8,
                        help="nms threshold")

    # logging
    parser.add_argument(
        "--log_checkpoints",
        action="store_true",
        help="Indicator to log intermediate model checkpoints",
    )
    parser.add_argument("--img_log_steps", type=int, default=500)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=
        ('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
         ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
         ),
    )

    # prompt
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<v>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument("--inference_prompt", default=None, type=str)

    # training
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help=
        "Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=
        ("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
         ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay",
                        type=float,
                        default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=
        ("Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
         " behaviors, so disable this argument if it causes any problems. More info:"
         " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
         ),
    )

    # data type
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
         ),
    )

    # for debugging (do not modify them, some of them will be overwritten)
    parser.add_argument(
        "--do_not_apply_masked_loss",
        action="store_false",
        help="Use masked loss instead of standard loss",
        dest="apply_masked_loss"
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_of_assets", 
        type=int, 
        default=2,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class CompCtrlPersDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        placeholder_tokens,
        tokenizer,
        size=512,
        flip_p=0.5,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.flip_p = flip_p

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.image_process = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if not Path(instance_data_root).exists():
            raise ValueError("Instance images root doesn't exists.")

        self.placeholder_tokens = placeholder_tokens

        # get paths for images and masks
        self.instance_images_path = [
            f for f in sorted(os.listdir(instance_data_root))
            if os.path.isfile(os.path.join(instance_data_root, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        mask_dir = os.path.join(instance_data_root, "masks")
        self.instance_masks_path = [
            f for f in sorted(os.listdir(mask_dir))
            if os.path.isfile(os.path.join(mask_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        assert len(self.instance_images_path) == len(self.instance_masks_path)

        # load images and masks
        self.instance_images = []
        self.instance_masks = []
        for i in range(len(self.instance_images_path)):
            # load and transform masks
            instance_mask_path = os.path.join(mask_dir,
                                              self.instance_masks_path[i])
            mask = Image.open(instance_mask_path)
            mask = self.mask_transforms(mask)[0, None, None, ...]
            self.instance_masks.append(mask)
            # load and transform images
            instance_image_path = os.path.join(instance_data_root,
                                               self.instance_images_path[i])
            image = Image.open(instance_image_path)
            image = self.image_transforms(image)
            self.instance_images.append(image)
        self.instance_images = torch.stack(self.instance_images)
        self.instance_masks = torch.cat(self.instance_masks)

        # get formatted prompts
        pair_name = os.path.basename(instance_data_root)
        sample_names_and_ids = pair_name.split('+')
        sample_names = [s.split('_')[0] for s in sample_names_and_ids]
        self.instance_prompts = []
        for p in self.instance_images_path:
            sample_name = p.split('_')[1]
            sample_idx = sample_names.index(sample_name)
            referent = self.placeholder_tokens[sample_idx]
            prompt = f"A photo of {referent}"
            self.instance_prompts.append(prompt)

        # load prompt indexs
        text_inputs = tokenize_prompt(
            self.tokenizer,
            self.instance_prompts,
        )
        self.instance_prompt_ids = text_inputs.input_ids

        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        example["instance_images"] = self.instance_images.clone()
        example["instance_masks"] = self.instance_masks.clone()
        example["instance_prompt_ids"] = self.instance_prompt_ids.clone()
        example["instance_images"] = self.image_process(example["instance_images"])
        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        return example


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]

    pixel_values = torch.cat(pixel_values, dim=0)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    masks = torch.cat(masks)
    masks = masks.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_masks": masks,
    }
    return batch


class MagicTailor:
    def __init__(self):
        self.args = parse_args()
        self.main()

    def main(self):

        # overwirtte args configs for a pair
        instance_images_path = [
            f for f in sorted(os.listdir(self.args.instance_data_dir))
            if os.path.isfile(os.path.join(self.args.instance_data_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.args.train_batch_size = len(instance_images_path)
        sample_names_and_ids = os.path.basename(self.args.instance_data_dir).split('+')
        self.args.num_of_assets = len(sample_names_and_ids)
        self.args.initializer_tokens = [s.split('_')[0] for s in sample_names_and_ids]

        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        accelerator_project_config = ProjectConfiguration(
           project_dir=self.args.output_dir, logging_dir=logging_dir)
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
            # logging_dir=logging_dir,
        )

        if (
            self.args.gradient_accumulation_steps > 1
            and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # if passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # text-guided mask generation
        if check_mask_existence(self.args.instance_data_dir):
            print("Masks alreadly exist.")
        else:
            print("Perform text-guided mask generation.")
            grounding_dino_model, sam_predictor = get_gdino_and_sam_model(self.args, self.accelerator.device)
            generate_masks(self.args, grounding_dino_model, sam_predictor, self.args.instance_data_dir, save_logs=True)
            del grounding_dino_model
            del sam_predictor

        # handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.args.pretrained_model_name_or_path, self.args.revision
        )

        # load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )

        # load the tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False
            )
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )

        # add placeholder tokens to tokenizer
        self.placeholder_tokens = [
            self.args.placeholder_token.replace(">", f"{idx}>")
            for idx in range(self.args.num_of_assets)
        ]
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
        assert num_added_tokens == self.args.num_of_assets
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.placeholder_tokens
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        if len(self.args.initializer_tokens) > 0:
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            for tkn_idx, initializer_token in enumerate(self.args.initializer_tokens):
                curr_token_ids = self.tokenizer.encode(
                    initializer_token, add_special_tokens=False
                )
                token_embeds[self.placeholder_token_ids[tkn_idx]] = token_embeds[
                    curr_token_ids[0]
                ].clone()
        else:
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            token_embeds[-self.args.num_of_assets :] = token_embeds[
                -3 * self.args.num_of_assets : -2 * self.args.num_of_assets
            ]

        # set validation scheduler for logging
        self.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.validation_scheduler.set_timesteps(50)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                print("Enable xformers.")
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.phase1_learning_rate = (
                self.args.phase1_learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )
            self.args.phase2_learning_rate = (
                self.args.phase2_learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # setup LoRA
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.2"],
        )
        self.unet.add_adapter(unet_lora_config)

        lora_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        params_to_optimize = (
            itertools.chain(
                lora_params,
                self.text_encoder.get_input_embeddings().parameters(),
            )
        )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.phase1_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # saving and loading ckpt for the model with LoRA
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                # there are only two options here. Either are just the unet attn processor layers
                # or there are the unet and text encoder atten layers
                unet_lora_layers_to_save = None
                text_encoder_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(self.unwrap_model(self.unet))):
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                    elif isinstance(model, type(self.unwrap_model(self.text_encoder))):
                        text_encoder_lora_layers_to_save = None
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            unet_ = None
            text_encoder_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(self.unwrap_model(self.unet))):
                    unet_ = model
                elif isinstance(model, type(self.unwrap_model(self.text_encoder))):
                    text_encoder_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

            unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            if self.args.mixed_precision == "fp16":
                models = [unet_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models, dtype=torch.float32)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        # create dataLoaders
        train_dataset = CompCtrlPersDataset(
            instance_data_root=self.args.instance_data_dir,
            placeholder_tokens=self.placeholder_tokens,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,  # load all images for once time, so here batch_size set to 1
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples),
            num_workers=self.args.dataloader_num_workers,
        )

        # math around the number of training steps
        # (nothing important here, you can just skip and keep this)
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
            * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (
            self.unet,
            self.text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # for mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # we need to recalculate our total training steps as the size of the training dataloader may have changed
        # (nothing important here, you can just skip and keep this)
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        if len(self.args.initializer_tokens) > 0:
            self.args.initializer_tokens = ", ".join(self.args.initializer_tokens)

        # we need to initialize the trackers we use, and also store our configuration
        # the trackers initializes automatically on the main process
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("MagicTailor", config=vars(self.args))

        # set the inference prompt is it is not given
        if self.args.inference_prompt == None:
            if len(self.placeholder_tokens) > 1:
                self.args.inference_prompt = f"{self.placeholder_tokens[0]} with " + " and ".join(
                    self.placeholder_tokens[1:]
                )
            else:
                self.args.inference_prompt = self.placeholder_tokens[0]

        # begin training
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Total number of reference images = {len(train_dataset)}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # get the mos recent checkpoint
                dirs = os.listdir(self.args.checkpoint_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.checkpoint_dir, path)) 
                global_step = int(path.split("-")[1])
                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )

        # only show the progress bar once on each machine
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # create the attention controller
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            for step, batch in enumerate(train_dataloader):

                if global_step == self.args.phase1_train_steps:
                    print("Warm-up ends. Switch to the DS-Bal Paradigm.")
                    # setup dual-streaming denoising U-Nets
                    # self.unet -> online denoising U-Net
                    # self.unet_m -> momentum denoising U-Net
                    self.unet_m_device = self.unet.device  # modify this to move unet_m to another GPU if you reach the GPU memory limit
                    self.unet_m = EMA(self.unet, decay=self.args.ema_decay, device=self.unet_m_device)
                    self.unet_m.requires_grad_(False)
                    # change lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.args.phase2_learning_rate

                logs = {}

                # skip steps until we reach the resumed step
                if (
                    self.args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                # core training code
                with self.accelerator.accumulate(self.unet):

                    # DM-Deg 
                    # adjust weight
                    curr_weight = self.args.alpha * (1 - ((global_step + 1) / self.args.max_train_steps) ** self.args.gamma)
                    # add noise
                    raw_noise = torch.randn_like(batch["pixel_values"])
                    masked_noise = raw_noise * (1 - batch["instance_masks"])
                    batch["pixel_values"] += curr_weight * masked_noise
                    batch["pixel_values"] = torch.clamp(batch["pixel_values"], -1, 1)

                    # convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"].to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    # sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # predict the noise residual
                    model_pred = self.unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    # get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )

                    # masked diffusion loss
                    if self.args.apply_masked_loss:
                        masks = batch["instance_masks"]
                        downsampled_masks = F.interpolate(input=masks,
                                                        size=(64, 64))
                        model_pred = model_pred * downsampled_masks
                        target = target * downsampled_masks

                    if global_step < self.args.phase1_train_steps:
                        # warm-up
                        diff_loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                        loss = diff_loss
                    else:
                        # DS-Bal
                        model_pred_m = self.unet_m(
                            noisy_latents.detach(), timesteps.detach(), encoder_hidden_states.detach()
                        ).sample
                        
                        # use the following one if self.unet_m and self.unet in different GPUs
                        # model_pred_m = self.unet_m(
                        #     noisy_latents.detach().to(self.unet_m_device),
                        #     timesteps.detach().to(self.unet_m_device),
                        #     encoder_hidden_states.detach().to(self.unet_m_device)
                        # ).sample.to(self.accelerator.device)

                        if self.args.apply_masked_loss:
                            model_pred_m = model_pred_m * downsampled_masks
                        sample_wise_shape = (self.args.num_of_assets, -1, *(target.shape[1:]))

                        # Sample-wise Min-Max Optimization
                        unet_loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        unet_loss = unet_loss.reshape(sample_wise_shape)
                        unet_loss = unet_loss.mean(dim=(1,2,3,4))
                        max_diff_loss = unet_loss.max()

                        # Selective Preserving Regularization
                        unet_m_loss = F.mse_loss(
                            model_pred.float(), model_pred_m.float(), reduction="none"
                        )
                        unet_m_loss = unet_m_loss.reshape(sample_wise_shape)
                        unet_m_loss = unet_m_loss.mean(dim=(1,2,3,4))
                        selected_idx = set(range(self.args.num_of_assets))
                        max_idx = torch.argmax(unet_loss).item()
                        selected_idx.discard(max_idx)
                        selected_idx = torch.tensor(list(selected_idx))
                        pres_loss = unet_m_loss[selected_idx].mean()

                        loss = max_diff_loss + self.args.lambda_preservation * pres_loss

                    # cross-attention loss
                    if self.args.lambda_attention != 0:
                        attn_loss = 0
                        losses_attn = []

                        GT_masks = F.interpolate(
                            input=batch["instance_masks"], size=(16, 16)
                        )
                        agg_attn = self.aggregate_attention(
                            res=16,
                            from_where=("up", "down"),
                            is_cross=True,
                        )

                        # set for curr_placeholder_token_id assignment with mask_id
                        self.serial_token_ids = [ 
                            int(f.split('_')[0]) for f in sorted(os.listdir(self.args.instance_data_dir))
                            if os.path.isfile(os.path.join(self.args.instance_data_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
                        ]

                        for mask_id in range(len(GT_masks)):

                            curr_placeholder_token_id = self.placeholder_token_ids[self.serial_token_ids[mask_id]]

                            curr_cond_batch_idx = mask_id  # set to this because mask num is equal to image num

                            asset_idx = (
                                (
                                    batch["input_ids"][curr_cond_batch_idx]
                                    == curr_placeholder_token_id
                                )
                                .nonzero()
                                .item()
                            )
                            # asset_attn_mask = agg_attn[..., asset_idx]
                            asset_attn_mask = agg_attn[mask_id, ..., asset_idx]
                            asset_attn_mask = (
                                asset_attn_mask / asset_attn_mask.max()  # normalize the attention mask
                            )
                            losses_attn.append(
                                F.mse_loss(
                                    GT_masks[mask_id, 0].float(),
                                    asset_attn_mask.float(),
                                    reduction="mean",
                                )
                            )

                        losses_attn = torch.stack(losses_attn)
                        attn_loss = losses_attn.mean()
                        loss = loss + self.args.lambda_attention * attn_loss

                    self.accelerator.backward(loss)

                    # no need to keep the attention store
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0

                    if self.accelerator.sync_gradients:
                        params_to_clip = (self.unet.parameters())
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.args.max_grad_norm
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

                    # update momentum denoising U-Net
                    if global_step >= self.args.phase1_train_steps:
                        self.unet_m.update_parameters(self.unet)

                # checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # save checkpoints
                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(
                                self.args.output_dir, f"checkpoint-{global_step}"
                            )
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                    # save images for logging
                    if ( 
                        self.args.log_checkpoints
                        and (global_step % self.args.img_log_steps == 0 or
                             global_step == self.args.max_train_steps)
                    ):
                        ckpts_path = os.path.join(
                            self.args.output_dir, "models", f"{global_step:05}"
                        )
                        os.makedirs(ckpts_path, exist_ok=True)
                        self.save_pipeline(ckpts_path)

                        img_logs_path = os.path.join(self.args.output_dir, "img_logs")
                        os.makedirs(img_logs_path, exist_ok=True)

                        if self.args.lambda_attention != 0:
                            self.controller.cur_step = 1

                            for mask_id in range(len(GT_masks)):
                                log_curr_cond_batch_idx = mask_id
                                log_sentence = batch["input_ids"][log_curr_cond_batch_idx]
                                log_sentence = log_sentence[
                                    (log_sentence != 0)
                                    & (log_sentence != 49406)
                                    & (log_sentence != 49407)
                                ]
                                log_sentence = self.tokenizer.decode(log_sentence)
                                self.save_cross_attention_vis(
                                    log_sentence,
                                    attention_maps=agg_attn[mask_id].detach().cpu(),
                                    path=os.path.join(
                                        img_logs_path, f"{global_step:05}_attn_{mask_id}.jpg"
                                    ),
                                )
                            self.controller.cur_step = 0
                            self.controller.attention_store = {}

                        self.perform_full_inference(
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_infer_img.jpg"
                            )
                        )

                        full_agg_attn = self.aggregate_attention(
                            res=16, from_where=("up", "down"), is_cross=True, is_inference=True
                        )
                        self.save_cross_attention_vis(
                            self.args.inference_prompt,
                            attention_maps=full_agg_attn.detach().cpu(),
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_infer_attn.jpg"
                            ),
                        )

                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                if global_step >= self.args.max_train_steps:
                    break

        self.save_pipeline(self.args.output_dir)

        self.accelerator.end_training()

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_pipeline(self, path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # saving LoRA weights
            unet = self.unwrap_model(self.unet)
            unet = unet.to(torch.float32)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            text_encoder_state_dict = None
            LoraLoaderMixin.save_lora_weights(
                save_directory=path,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
            )
            # saving token embeddings
            torch.save(
                self.text_encoder.get_input_embeddings().state_dict(),
                os.path.join(path, 'token_embedding.pth'),
            )
            # saving the tokenizer
            self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
        self, res: int, from_where: List[str], is_cross: bool, is_inference=False,
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    if is_inference:
                        cross_maps = item.reshape(
                            2, -1, res, res, item.shape[-1]
                        )[1]
                    else:                        
                        cross_maps = item.reshape(
                            self.args.train_batch_size, -1, res, res, item.shape[-1]
                        )
                    out.append(cross_maps)

        if is_inference:
            out = torch.cat(out, dim=0)
            out = out.sum(0) / out.shape[0]
        else:
            out = torch.cat(out, dim=1)
            out = out.sum(1) / out.shape[1]

        return out

    @torch.no_grad()
    def perform_full_inference(self, path, guidance_scale=7.5):
        self.unet.eval()
        self.text_encoder.eval()

        latents = torch.randn((1, 4, 64, 64), device=self.accelerator.device)
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.accelerator.device)
        input_ids = self.tokenizer(
            self.args.inference_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        cond_embeddings = self.text_encoder(input_ids)[0]
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.vae.decode(latents.to(self.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.unet.train()

        Image.fromarray(images[0]).save(path)

    def make_image_grid(self, images, rows, cols, resize=None):
        """
        Prepares a single grid of images. Useful for visualization purposes.
        """
        assert len(images) == rows * cols
        if resize is not None:
            images = [img.resize((resize, resize)) for img in images]
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path):
        tokens = self.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            if int(tokens[i]) in [0, 49406, 49407]:
                continue
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((512, 512)))
            image = image[:, :, ::-1].copy()
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            image = image[:, :, ::-1].copy()
            image = ptp_utils.text_under_image(
                image, self.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = ptp_utils.view_images(np.stack(images, axis=0))
        vis.save(path)


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class EMA(torch.optim.swa_utils.AveragedModel):
    """
    Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg, use_buffers=True)


if __name__ == "__main__":
    MagicTailor()
