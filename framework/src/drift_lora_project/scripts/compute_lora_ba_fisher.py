#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Callable, Optional

import torch
from safetensors.torch import save_file

from run_swift_sft_ewc import normalize_key
from swift.pipelines.train.sft import SwiftSft
from swift.trainers import TrainerFactory
from swift.utils import get_logger, is_master

try:
    from peft.tuners.lora.layer import VARIANT_KWARG_KEYS
except Exception:
    VARIANT_KWARG_KEYS = []


logger = get_logger()


class FisherSft(SwiftSft):
    def build_trainer(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        args.save_args()
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            template=self.template,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            **self._get_trainer_kwargs(),
        )
        return trainer


class BaGradCapture:
    def __init__(self) -> None:
        self.grads: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        self.grads.clear()

    def hook(self, key: str) -> Callable[[torch.Tensor], None]:
        def save_grad(grad: torch.Tensor) -> None:
            self.grads[key] = grad.detach()

        return save_grad


def active_lora_adapter(module: torch.nn.Module) -> Optional[str]:
    lora_a = getattr(module, "lora_A", None)
    if lora_a is None:
        return None
    active_adapters = getattr(module, "active_adapters", None)
    if active_adapters:
        for adapter in active_adapters:
            if adapter in lora_a:
                return adapter
    keys = list(lora_a.keys())
    return keys[0] if keys else None


def is_target_module(module_key: str, targets: tuple[str, ...]) -> bool:
    return any(module_key.endswith(f".{target}") or module_key.endswith(target) for target in targets)


def make_strict_ba_forward(
    module_key: str,
    original_forward: Callable,
    capture: BaGradCapture,
) -> Callable:
    def strict_ba_forward(self, x: torch.Tensor, *args, **kwargs):
        original_kwargs = dict(kwargs)
        if getattr(self, "disable_adapters", False) or getattr(self, "merged", False):
            return original_forward(x, *args, **original_kwargs)
        if original_kwargs.get("adapter_names") is not None:
            return original_forward(x, *args, **original_kwargs)

        active_adapters = list(getattr(self, "active_adapters", []) or [])
        lora_a_keys = getattr(self, "lora_A", {}).keys()
        if not active_adapters or not any(adapter in lora_a_keys for adapter in active_adapters):
            return original_forward(x, *args, **original_kwargs)
        lora_variant = getattr(self, "lora_variant", {})
        if any(adapter in lora_variant for adapter in active_adapters):
            return original_forward(x, *args, **original_kwargs)

        self._check_forward_args(x, *args, **kwargs)
        kwargs = dict(kwargs)
        kwargs.pop("adapter_names", None)
        for key in VARIANT_KWARG_KEYS:
            kwargs.pop(key, None)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for active_adapter in active_adapters:
            if active_adapter not in self.lora_A:
                continue

            lora_a = self.lora_A[active_adapter]
            lora_b = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = float(self.scaling[active_adapter])

            x_lora = self._cast_input_dtype(x, lora_a.weight.dtype)
            dropped = dropout(x_lora)
            delta_w = lora_b.weight @ lora_a.weight
            if scaling != 1.0:
                delta_w = delta_w * scaling
            delta_w = delta_w.detach().requires_grad_(True)
            delta_w.register_hook(capture.hook(module_key))
            update = dropped @ delta_w.transpose(0, 1)
            if lora_b.bias is not None:
                update = update + lora_b.bias.to(update.dtype) * scaling
            result = result + update

        return result.to(torch_result_dtype)

    return strict_ba_forward


def install_strict_ba_forwards(
    model: torch.nn.Module,
    capture: BaGradCapture,
    targets: tuple[str, ...],
) -> list[str]:
    patched: list[str] = []
    for module_name, module in model.named_modules():
        module_key = normalize_key(module_name)
        if not is_target_module(module_key, targets):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if active_lora_adapter(module) is None:
            continue
        original_forward = module.forward
        module.forward = types.MethodType(make_strict_ba_forward(module_key, original_forward, capture), module)
        patched.append(module_key)
    return patched


def prepare_model_for_fisher(model: torch.nn.Module) -> None:
    # Keep checkpointing active for large VLM batches while avoiding dropout noise in the Fisher estimate.
    model.train()
    trainable_before = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_before += parameter.numel()
        parameter.requires_grad_(False)

    dropout_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
            dropout_count += 1

    config_count = 0
    seen_configs: set[int] = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        if hasattr(config, "use_cache"):
            config.use_cache = False
            config_count += 1

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = False

    logger.info(
        "Prepared Fisher model: train mode for gradient checkpointing, "
        f"froze trainable params={trainable_before}, dropout modules kept eval={dropout_count}, "
        f"use_cache disabled configs={config_count}"
    )


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Compute dense BA-space Fisher for selected LoRA modules using Swift.")
    parser.add_argument("--fisher-output", required=True, type=Path)
    parser.add_argument("--fisher-summary", type=Path, default=None)
    parser.add_argument("--fisher-max-batches", type=int, default=0)
    parser.add_argument("--fisher-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--fisher-accumulation", choices=["cpu", "gpu_local"], default="gpu_local")
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--fisher-scale-by-batch", dest="fisher_scale_by_batch", action="store_true", default=True)
    scale_group.add_argument("--no-fisher-scale-by-batch", dest="fisher_scale_by_batch", action="store_false")
    parser.add_argument("--ba-target-modules", default="k_proj,v_proj")
    return parser.parse_known_args(argv)


def cast_for_save(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    tensor = tensor.contiguous()
    if dtype == "float16":
        return tensor.half()
    if dtype == "bfloat16":
        return tensor.bfloat16()
    return tensor.float()


def main() -> None:
    custom_args, swift_args = parse_args(sys.argv[1:])
    targets = tuple(item.strip() for item in custom_args.ba_target_modules.split(",") if item.strip())
    pipeline = FisherSft(swift_args)
    trainer = pipeline.build_trainer()
    model = trainer.model
    dataloader = trainer.get_train_dataloader()
    prepare_model_for_fisher(model)

    capture = BaGradCapture()
    patched_modules = install_strict_ba_forwards(model, capture, targets)
    if not patched_modules:
        raise RuntimeError(f"No LoRA modules were patched for BA Fisher. targets={targets}")
    logger.info(f"Strict BA Fisher patched modules: {len(patched_modules)} targets={','.join(targets)}")

    fisher: dict[str, torch.Tensor] = {}
    num_batches = 0
    num_examples = 0

    for batch in dataloader:
        if custom_args.fisher_max_batches and num_batches >= custom_args.fisher_max_batches:
            break
        capture.clear()
        model.zero_grad(set_to_none=True)
        prepared = trainer._prepare_inputs(batch)
        labels = prepared.get("labels")
        batch_size = int(labels.shape[0]) if isinstance(labels, torch.Tensor) and labels.ndim > 0 else 1
        loss = trainer.compute_loss(model, prepared)
        trainer.accelerator.backward(loss)
        weight = batch_size if custom_args.fisher_scale_by_batch else 1

        for key, grad in capture.grads.items():
            grad2 = grad.float().square()
            if weight != 1:
                grad2 = grad2 * weight
            if custom_args.fisher_accumulation == "cpu":
                grad2 = grad2.cpu()
            if key in fisher:
                fisher[key].add_(grad2.to(fisher[key].device))
            else:
                fisher[key] = grad2

        num_batches += 1
        num_examples += batch_size
        if num_batches % 10 == 0:
            logger.info(
                f"Strict BA Fisher progress: batches={num_batches} examples={num_examples} "
                f"tensors={len(fisher)}/{len(patched_modules)}"
            )

    denom = max(num_examples if custom_args.fisher_scale_by_batch else num_batches, 1)
    save_fisher: dict[str, torch.Tensor] = {}
    for key in list(fisher):
        tensor = fisher.pop(key)
        tensor.div_(denom)
        save_fisher[key] = cast_for_save(tensor, custom_args.fisher_dtype).cpu()
        del tensor

    if is_master():
        custom_args.fisher_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(save_fisher, str(custom_args.fisher_output))
        summary = {
            "fisher_output": str(custom_args.fisher_output),
            "mode": "ba_kv",
            "ba_target_modules": list(targets),
            "patched_modules": len(patched_modules),
            "num_tensors": len(save_fisher),
            "num_batches": num_batches,
            "num_examples": num_examples,
            "dtype": custom_args.fisher_dtype,
            "accumulation": custom_args.fisher_accumulation,
            "scale_by_batch": bool(custom_args.fisher_scale_by_batch),
            "total_elements": int(sum(t.numel() for t in save_fisher.values())),
        }
        summary_path = custom_args.fisher_summary or custom_args.fisher_output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
