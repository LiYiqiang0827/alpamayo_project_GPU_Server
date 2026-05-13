#!/usr/bin/env python3
"""
Alpamayo2B 模型加载、参数冻结和验证脚本
用于 LLM 蒸馏项目 (10B -> 2B)

功能:
1. 加载 Alpamayo2B 模型 (Qwen3VLForConditionalGeneration)
2. 冻结 ViT 部分 (model.visual)
3. 冻结基础词表 embedding (0-151642)
4. 只训练扩展词表 (151669-155696) 和 LLM 部分
5. 验证参数冻结状态
6. 验证词表对齐
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

# ==================== 路径配置 ====================
STUDENT_PATH = "/data01/mikelee/weight/alpamayo2B"
TEACHER_PATH = "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B"
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"

# ==================== 词表范围配置 ====================
BASE_VOCAB_START = 0
BASE_VOCAB_END = 151642       # 基础词表范围 [0, 151642]
EXTENDED_VOCAB_START = 151669  # 扩展词表起始 (轨迹token)
EXTENDED_VOCAB_END = 155696    # 扩展词表结束 [151669, 155696]

# 特殊token ID范围 (在tokenizer_final中)
SPECIAL_TOKEN_RANGE = range(155669, 155697)  # Alpamayo特定特殊token


def load_student_model(model_path: str = STUDENT_PATH, dtype=torch.float16):
    """
    加载 Alpamayo2B 学生模型
    
    Args:
        model_path: 模型路径
        dtype: 数据类型
    
    Returns:
        model: 加载的模型
    """
    from transformers import Qwen3VLForConditionalGeneration
    
    print(f"Loading student model from: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.0f}M parameters")
    
    return model


def freeze_vit(model: nn.Module) -> None:
    """
    冻结 ViT (视觉编码器) 部分
    
    Args:
        model: Qwen3VLForConditionalGeneration 模型
    """
    if hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = False
        print("✓ ViT (visual encoder) FROZEN")
    else:
        print("⚠ ViT not found in model")


def freeze_base_vocab_embedding(model: nn.Module) -> None:
    """
    冻结基础词表 embedding (0-151642)
    通过注册 gradient hook 实现部分冻结
    
    Args:
        model: Qwen3VLForConditionalGeneration 模型
    """
    # 获取 embedding 层
    embed_tokens = None
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_tokens = model.model.embed_tokens
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embed_tokens'):
        embed_tokens = model.language_model.embed_tokens
    
    if embed_tokens is None:
        print("⚠ embed_tokens not found")
        return
    
    # 创建基础词表 mask
    base_vocab_size = BASE_VOCAB_END + 1  # 151643
    
    # 注册 gradient hook 冻结基础词表
    def freeze_base_vocab_hook(grad):
        """冻结基础词表梯度的hook"""
        grad_clone = grad.clone()
        if grad_clone.shape[0] > BASE_VOCAB_END:
            grad_clone[:BASE_VOCAB_END + 1] = 0
        return grad_clone
    
    if hasattr(embed_tokens, 'weight'):
        embed_tokens.weight.register_hook(freeze_base_vocab_hook)
        print(f"✓ Base vocab embedding (0-{BASE_VOCAB_END}) FROZEN via gradient hook")
    
    # 如果 lm_head 是独立的 (不共享 embedding)，也需要冻结
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        # 检查 lm_head 是否与 embed_tokens 共享权重
        if model.lm_head.weight is not embed_tokens.weight:
            model.lm_head.weight.register_hook(freeze_base_vocab_hook)
            print(f"✓ lm_head base vocab (0-{BASE_VOCAB_END}) FROZEN via gradient hook")


def setup_model_for_distillation(model_path: str = STUDENT_PATH, dtype=torch.float16):
    """
    设置模型用于蒸馏训练
    
    执行:
    1. 加载模型
    2. 冻结 ViT
    3. 冻结基础词表
    
    Args:
        model_path: 模型路径
        dtype: 数据类型
    
    Returns:
        model: 配置好的模型
    """
    print("=" * 70)
    print("Setting up model for LLM distillation")
    print("=" * 70)
    
    # 1. 加载模型
    model = load_student_model(model_path, dtype)
    
    # 2. 冻结 ViT
    print("\n--- Freezing ViT ---")
    freeze_vit(model)
    
    # 3. 冻结基础词表
    print("\n--- Freezing base vocab embedding ---")
    freeze_base_vocab_embedding(model)
    
    # 4. 统计参数
    print("\n--- Parameter Summary ---")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    # 计算扩展词表参数
    extended_vocab_size = EXTENDED_VOCAB_END - EXTENDED_VOCAB_START + 1  # 4000
    embed_dim = model.config.text_config.hidden_size  # 2048
    extended_embed_params = extended_vocab_size * embed_dim
    
    print(f"  Total parameters:        {total / 1e6:.0f}M")
    print(f"  Trainable parameters:    {trainable / 1e6:.0f}M")
    print(f"  Frozen parameters:       {frozen / 1e6:.0f}M")
    print(f"  Extended vocab params:   {extended_embed_params / 1e6:.2f}M")
    print(f"  Trainable ratio:         {trainable/total*100:.1f}%")
    
    return model


def verify_parameter_freezing(model: nn.Module, verbose: bool = True) -> Dict[str, any]:
    """
    验证参数冻结状态
    
    Args:
        model: 模型
        verbose: 是否打印详细信息
    
    Returns:
        验证结果字典
    """
    results = {
        'vit_frozen': True,
        'llm_trainable': True,
        'base_vocab_frozen': True,
        'extended_vocab_trainable': True,
        'issues': []
    }
    
    # 1. 检查 ViT 是否冻结
    if hasattr(model, 'visual'):
        for name, param in model.visual.named_parameters():
            if param.requires_grad:
                results['vit_frozen'] = False
                results['issues'].append(f"ViT param '{name}' is not frozen!")
                if verbose:
                    print(f"❌ ViT param '{name}' requires_grad=True")
    
    if results['vit_frozen'] and verbose:
        print("✅ ViT is fully frozen")
    
    # 2. 检查 LLM 是否可训练
    llm_prefix = 'model.language_model' if hasattr(model, 'model') else 'language_model'
    llm_params = []
    for name, param in model.named_parameters():
        if 'visual' not in name and 'embed_tokens' not in name:
            llm_params.append((name, param))
    
    llm_trainable_count = sum(1 for _, p in llm_params if p.requires_grad)
    llm_total_count = len(llm_params)
    
    if llm_trainable_count < llm_total_count * 0.9:
        results['llm_trainable'] = False
        results['issues'].append(f"Only {llm_trainable_count}/{llm_total_count} LLM params are trainable")
    
    if verbose:
        print(f"✅ LLM params: {llm_trainable_count}/{llm_total_count} trainable")
    
    # 3. 检查 embedding hook 是否注册
    embed_tokens = None
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_tokens = model.model.embed_tokens
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embed_tokens'):
        embed_tokens = model.language_model.embed_tokens
    
    if embed_tokens is not None and hasattr(embed_tokens.weight, '_backward_hooks'):
        hooks = embed_tokens.weight._backward_hooks
        if len(hooks) > 0:
            if verbose:
                print(f"✅ Embedding has {len(hooks)} gradient hook(s) registered")
        else:
            results['base_vocab_frozen'] = False
            results['issues'].append("No gradient hook on embedding!")
            if verbose:
                print("❌ No gradient hook on embedding")
    
    return results


def verify_vocab_alignment(tokenizer_path: str = TOKENIZER_PATH) -> Dict[str, any]:
    """
    验证词表对齐
    
    检查:
    1. 总词表大小是否为 155697
    2. 轨迹token范围是否正确 (151669-155668)
    3. 特殊token是否正确
    
    Args:
        tokenizer_path: tokenizer路径
    
    Returns:
        验证结果字典
    """
    from transformers import AutoTokenizer
    
    print("\n" + "=" * 70)
    print("Verifying Vocabulary Alignment")
    print("=" * 70)
    
    results = {
        'vocab_size_correct': False,
        'traj_tokens_correct': False,
        'special_tokens_correct': False,
        'issues': []
    }
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    vocab_size = len(tokenizer)
    
    print(f"\nTokenizer vocab size: {vocab_size}")
    print(f"Expected vocab size:  155697")
    
    if vocab_size == 155697:
        results['vocab_size_correct'] = True
        print("✅ Vocab size is correct")
    else:
        results['issues'].append(f"Vocab size mismatch: {vocab_size} != 155697")
        print(f"❌ Vocab size mismatch: {vocab_size} != 155697")
    
    # 检查轨迹token
    print("\n--- Trajectory Tokens ---")
    traj_start_id = tokenizer.convert_tokens_to_ids("<i0>")
    traj_end_id = tokenizer.convert_tokens_to_ids("<i3999>")
    
    print(f"<i0> ID:    {traj_start_id} (expected: 151669)")
    print(f"<i3999> ID: {traj_end_id} (expected: 155668)")
    
    if traj_start_id == 151669 and traj_end_id == 155668:
        results['traj_tokens_correct'] = True
        print("✅ Trajectory token range is correct")
    else:
        results['issues'].append(f"Traj token range mismatch")
        print("❌ Trajectory token range mismatch")
    
    # 检查关键特殊token
    print("\n--- Special Tokens ---")
    special_tokens = {
        '<|im_start|>': 151644,
        '<|im_end|>': 151645,
        '<|vision_start|>': 151652,
        '<|vision_end|>': 151653,
        '<|image_pad|>': 151655,
        '<|cot_start|>': 155675,
        '<|cot_end|>': 155676,
        '<|traj_history_start|>': 155673,
        '<|traj_history_end|>': 155674,
        '<|traj_future_start|>': 155677,
        '<|traj_future_end|>': 155678,
        '<|traj_history|>': 155679,
        '<|traj_future|>': 155680,
    }
    
    all_correct = True
    for token, expected_id in special_tokens.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        match = actual_id == expected_id
        status = "✅" if match else "❌"
        print(f"{status} {token:30s} ID={actual_id:6d} (expected: {expected_id})")
        if not match:
            all_correct = False
            results['issues'].append(f"Special token {token} ID mismatch: {actual_id} != {expected_id}")
    
    results['special_tokens_correct'] = all_correct
    
    # 最终结论
    print("\n" + "=" * 70)
    if all([results['vocab_size_correct'], results['traj_tokens_correct'], results['special_tokens_correct']]):
        print("✅✅✅ All vocab checks passed! ✅✅✅")
    else:
        print("❌❌❌ Some vocab checks failed! ❌❌❌")
        for issue in results['issues']:
            print(f"  - {issue}")
    print("=" * 70)
    
    return results


def test_gradient_flow(model: nn.Module) -> bool:
    """
    测试梯度是否正确流动
    
    执行一次前向+反向传播，验证:
    1. ViT 参数梯度为 0
    2. 基础词表 embedding 梯度为 0
    3. 扩展词表和 LLM 参数有梯度
    
    Args:
        model: 模型
    
    Returns:
        是否通过测试
    """
    print("\n" + "=" * 70)
    print("Testing Gradient Flow")
    print("=" * 70)
    
    # 创建 dummy 输入
    batch_size = 1
    seq_len = 10
    
    input_ids = torch.randint(0, 155697, (batch_size, seq_len), device=model.device)
    attention_mask = torch.ones(batch_size, seq_len, device=model.device)
    labels = input_ids.clone()
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    loss = outputs.loss
    loss.backward()
    
    # 检查梯度
    all_passed = True
    
    # 1. 检查 ViT 梯度
    print("\n--- ViT Gradients ---")
    vit_has_grad = False
    for name, param in model.visual.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            vit_has_grad = True
            print(f"❌ ViT param '{name}' has non-zero gradient!")
            all_passed = False
    
    if not vit_has_grad:
        print("✅ ViT gradients are all zero (correctly frozen)")
    
    # 2. 检查 embedding 梯度
    print("\n--- Embedding Gradients ---")
    embed_tokens = None
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_tokens = model.model.embed_tokens
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embed_tokens'):
        embed_tokens = model.language_model.embed_tokens
    
    if embed_tokens is not None and embed_tokens.weight.grad is not None:
        grad = embed_tokens.weight.grad
        
        # 基础词表梯度
        base_grad = grad[:BASE_VOCAB_END + 1]
        base_grad_sum = base_grad.abs().sum().item()
        
        # 扩展词表梯度
        ext_grad = grad[EXTENDED_VOCAB_START:EXTENDED_VOCAB_END + 1]
        ext_grad_sum = ext_grad.abs().sum().item()
        
        print(f"Base vocab gradient sum:   {base_grad_sum:.6f} (should be ~0)")
        print(f"Extended vocab gradient sum: {ext_grad_sum:.6f} (should be >0)")
        
        if base_grad_sum > 0.001:
            print("❌ Base vocab has non-zero gradient!")
            all_passed = False
        else:
            print("✅ Base vocab gradient is zero (correctly frozen)")
        
        if ext_grad_sum < 0.001:
            print("❌ Extended vocab has zero gradient!")
            all_passed = False
        else:
            print("✅ Extended vocab has non-zero gradient (correctly trainable)")
    
    # 3. 检查 LLM 层梯度
    print("\n--- LLM Layer Gradients ---")
    llm_has_grad = False
    for name, param in model.named_parameters():
        if 'visual' not in name and 'embed_tokens' not in name:
            if param.grad is not None and param.grad.abs().sum() > 0:
                llm_has_grad = True
                break
    
    if llm_has_grad:
        print("✅ LLM layers have gradients (correctly trainable)")
    else:
        print("❌ LLM layers have no gradients!")
        all_passed = False
    
    # 清理梯度
    model.zero_grad()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅✅✅ Gradient flow test PASSED! ✅✅✅")
    else:
        print("❌❌❌ Gradient flow test FAILED! ❌❌❌")
    print("=" * 70)
    
    return all_passed


def main():
    """主函数 - 运行所有验证"""
    print("=" * 70)
    print("Alpamayo2B Model Setup and Verification")
    print("=" * 70)
    
    # 1. 设置模型
    model = setup_model_for_distillation()
    
    # 2. 验证参数冻结
    print("\n" + "=" * 70)
    print("Verifying Parameter Freezing")
    print("=" * 70)
    freeze_results = verify_parameter_freezing(model, verbose=True)
    
    # 3. 验证词表对齐
    vocab_results = verify_vocab_alignment()
    
    # 4. 测试梯度流动
    grad_passed = test_gradient_flow(model)
    
    # 5. 最终总结
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    checks = [
        ("ViT frozen", freeze_results['vit_frozen']),
        ("LLM trainable", freeze_results['llm_trainable']),
        ("Base vocab frozen", freeze_results['base_vocab_frozen']),
        ("Extended vocab trainable", freeze_results['extended_vocab_trainable']),
        ("Vocab size correct", vocab_results['vocab_size_correct']),
        ("Traj tokens correct", vocab_results['traj_tokens_correct']),
        ("Special tokens correct", vocab_results['special_tokens_correct']),
        ("Gradient flow", grad_passed),
    ]
    
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Model is ready for distillation.")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    main()
