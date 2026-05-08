# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Alpamayo2B - 基于CosmosReason2-2B的自动驾驶VLM模型

核心特点：
- VLM Backbone: Qwen3-VL-2B (28层, 2048维)
- 输入: 16张图片 + 历史轨迹 + 导航指令
- 输出: CoT推理文本 + 轨迹预测
- 无Action Expert (仅VLM部分)
"""

import copy
import logging
from typing import Any, Optional

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)


# Alpamayo1.5的特殊token定义
SPECIAL_TOKENS = {
    "cot_start": "<|cot_start|>",
    "cot_end": "<|cot_end|>",
    "traj_history_start": "<|traj_history_start|>",
    "traj_history": "<|traj_history|>",
    "traj_history_end": "<|traj_history_end|>",
    "traj_future_start": "<|traj_future_start|>",
    "traj_future": "<|traj_future|>",
    "traj_future_end": "<|traj_future_end|>",
    "route_start": "<|route_start|>",
    "route_pad": "<|route_pad|>",
    "route_end": "<|route_end|>",
    "question_start": "<|question_start|>",
    "question_end": "<|question_end|>",
    "answer_start": "<|answer_start|>",
    "answer_end": "<|answer_end|>",
    "prompt_start": "<|prompt_start|>",
    "prompt_end": "<|prompt_end|>",
}

TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}


class Alpamayo2BConfig(PretrainedConfig):
    """Alpamayo2B配置类"""
    
    model_type = "alpamayo2b"
    
    def __init__(
        self,
        vlm_name_or_path: str = "nvidia/Cosmos-Reason2-2B",
        vocab_size: int = 155714,
        traj_vocab_size: int = 4000,
        traj_token_start_idx: int = 151669,
        tokens_per_history_traj: int = 48,
        tokens_per_future_traj: int = 128,
        min_pixels: int = 163840,
        max_pixels: int = 196608,
        add_special_tokens: bool = True,
        model_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ):
        self.vlm_name_or_path = vlm_name_or_path
        self.vocab_size = vocab_size
        self.traj_vocab_size = traj_vocab_size
        self.traj_token_start_idx = traj_token_start_idx
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.add_special_tokens = add_special_tokens
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation
        
        super().__init__(**kwargs)


class Alpamayo2B(PreTrainedModel):
    """
    Alpamayo2B: 自动驾驶视觉语言模型 (2B参数)
    
    基于CosmosReason2-2B (Qwen3-VL-2B) 架构，扩展了：
    - 轨迹token (4000个离散bin)
    - 特殊token (CoT, 轨迹标记, 导航标记等)
    - 自动驾驶特定的chat template
    """
    
    config_class = Alpamayo2BConfig
    base_model_prefix = "vlm"
    
    def __init__(
        self,
        config: Alpamayo2BConfig,
        pretrained_vlm: Optional[Qwen3VLForConditionalGeneration] = None,
    ):
        super().__init__(config)
        
        self.config = config
        
        # 初始化VLM backbone
        if pretrained_vlm is not None:
            self.vlm = pretrained_vlm
            logger.info("Using provided pretrained VLM")
        else:
            self._initialize_vlm(config)
        
        # 构建tokenizer
        self.tokenizer = self._build_tokenizer(config)
        self.processor = self._build_processor(config)
        
        # 记录token ID映射
        self.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v) 
            for k, v in SPECIAL_TOKENS.items()
        }
        self.traj_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v)
            for k, v in TRAJ_TOKEN.items()
        }
        
        # 记录参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _initialize_vlm(self, config: Alpamayo2BConfig) -> None:
        """初始化Qwen3-VL-2B VLM backbone"""
        logger.info(f"Initializing VLM from: {config.vlm_name_or_path}")
        
        vlm_config = Qwen3VLConfig.from_pretrained(
            config.vlm_name_or_path,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
        )
        
        # 设置vocab_size (已扩展)
        vlm_config.text_config.vocab_size = config.vocab_size
        vlm_config.vocab_size = config.vocab_size
        
        self.vlm = Qwen3VLForConditionalGeneration(vlm_config)
        logger.info(f"VLM initialized: {vlm_config.text_config.num_hidden_layers} layers, "
                    f"hidden_size={vlm_config.text_config.hidden_size}")
    
    def _build_tokenizer(self, config: Alpamayo2BConfig) -> AutoTokenizer:
        """构建tokenizer，添加轨迹token和特殊token"""
        processor_kwargs = {}
        if config.min_pixels is not None:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels is not None:
            processor_kwargs["max_pixels"] = config.max_pixels
        
        processor = AutoProcessor.from_pretrained(
            config.vlm_name_or_path, 
            **processor_kwargs,
            trust_remote_code=True,
        )
        tokenizer = processor.tokenizer
        
        # 添加轨迹token
        if config.traj_vocab_size is not None:
            discrete_tokens = [f"<|i{v}|>" for v in range(config.traj_vocab_size)]
            tokenizer.add_tokens(discrete_tokens)
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<|i0|>")
        
        # 添加特殊token
        if config.add_special_tokens:
            tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
        else:
            tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)
        
        # 记录轨迹token ID
        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v)
            for k, v in TRAJ_TOKEN.items()
        }
        
        return tokenizer
    
    def _build_processor(self, config: Alpamayo2BConfig) -> AutoProcessor:
        """构建processor"""
        processor_kwargs = {}
        if config.min_pixels is not None:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels is not None:
            processor_kwargs["max_pixels"] = config.max_pixels
        
        processor = AutoProcessor.from_pretrained(
            config.vlm_name_or_path,
            **processor_kwargs,
            trust_remote_code=True,
        )
        
        return processor
    
    def freeze_vit(self):
        """冻结Vision Encoder"""
        logger.info("Freezing Vision Encoder...")
        for param in self.vlm.visual.parameters():
            param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable parameters: {trainable / 1e6:.0f}M / {total / 1e6:.0f}M "
                    f"({trainable/total*100:.1f}%)")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """前向传播"""
        return self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """生成文本"""
        return self.vlm.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            **kwargs,
        )
    
    def prepare_inputs_for_generation(
        self,
        images,
        history_traj=None,
        nav_text=None,
        cot_prompt=None,
    ):
        """
        准备生成输入
        
        Args:
            images: 16张图片 (PIL Image列表)
            history_traj: 历史轨迹数据
            nav_text: 导航指令文本
            cot_prompt: CoT提示文本
        
        Returns:
            model_inputs: 可直接传入模型的输入
        """
        # 构建prompt
        image_placeholders = ""
        for _ in images:
            image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>"
        
        # 轨迹占位符
        traj_text = "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
        
        # 导航指令
        route_text = ""
        if nav_text:
            route_text = f"<|route_start|>{nav_text}<|route_end|>"
        
        # 构建messages
        messages = [
            {
                "role": "system",
                "content": "You are a driving assistant that generates safe and accurate actions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{image_placeholders}{traj_text}{route_text}output the chain-of-thought reasoning of the driving process, then output the future trajectory."}
                ]
            },
        ]
        
        if cot_prompt:
            messages.append({
                "role": "assistant",
                "content": f"<|cot_start|>{cot_prompt}"
            })
        
        # 应用chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not cot_prompt,
        )
        
        # 处理输入
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        return inputs
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """保存模型"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存VLM
        self.vlm.save_pretrained(save_directory, **kwargs)
        
        # 保存tokenizer和processor
        self.tokenizer.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        
        # 保存配置
        self.config.save_pretrained(save_directory)
        
        logger.info(f"Model saved to: {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """从预训练路径加载模型"""
        # 加载配置
        config = Alpamayo2BConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # 加载VLM
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        
        # 创建模型实例
        model = cls(config, pretrained_vlm=vlm)
        
        return model
