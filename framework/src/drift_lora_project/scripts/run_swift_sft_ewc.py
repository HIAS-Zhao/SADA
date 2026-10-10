#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from swift.pipelines.train.sft import SwiftSft
from swift.trainers.utils import per_token_loss_func
from swift.utils import get_logger


logger = get_logger()


def _as_bool(text: str) -> bool:
    return text.lower() in {"1", "true", "yes", "y", "on"}


def normalize_key(key: str) -> str:
    key = key.removeprefix("module.")
    key = key.replace(".default.", ".")
    return key


def find_adapter_file(path_text: str) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_file():
        return path
    candidate = path / "adapter_model.safetensors"
    if candidate.is_file():
        return candidate
    return None


def load_safetensor_map(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[normalize_key(key)] = handle.get_tensor(key).detach().cpu()
    return tensors


class SwiftEWC:
    def __init__(self) -> None:
        self.mode = os.environ.get("EWC_MODE", "lora_param").strip()
        self.enabled = self.mode not in {"", "none", "off", "0"}
        self.anchor_mode = os.environ.get("EWC_ANCHOR_MODE", "adapter").strip()
        self.weight = float(os.environ.get("EWC_LAMBDA", "0"))
        self.reduction = os.environ.get("EWC_REDUCTION", "mean").strip()
        self.normalize = os.environ.get("EWC_NORMALIZE", "mean").strip()
        self.include_regex_text = os.environ.get("EWC_INCLUDE_REGEX", "")
        self.cache_enabled = _as_bool(os.environ.get("EWC_CACHE_TENSORS", "1"))
        self.warn_missing = _as_bool(os.environ.get("EWC_WARN_MISSING", "1"))
        self.apply_half = _as_bool(os.environ.get("EWC_APPLY_HALF", "1"))
        self.ba_targets = tuple(
            item.strip() for item in os.environ.get("EWC_BA_TARGET_MODULES", "k_proj,v_proj").split(",") if item.strip()
        )

        self.anchor: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}
        self.cache: dict[tuple[str, str, torch.device, torch.dtype], torch.Tensor] = {}
        self._missing: set[str] = set()
        self._reported = False

        if not self.enabled:
            return
        if self.mode not in {"lora_param", "ba_kv"}:
            raise ValueError(
                f"EWC_MODE={self.mode!r} is not implemented in this Swift runner. "
                "Use EWC_MODE=lora_param or EWC_MODE=ba_kv."
            )
        if self.anchor_mode not in {"adapter", "initial_zero"}:
            raise ValueError(
                f"EWC_ANCHOR_MODE={self.anchor_mode!r} is not implemented. "
                "Use EWC_ANCHOR_MODE=adapter or EWC_ANCHOR_MODE=initial_zero."
            )

        fisher_path = Path(os.environ["EWC_FISHER_STATE"])
        if not fisher_path.is_file():
            raise FileNotFoundError(f"Missing EWC_FISHER_STATE: {fisher_path}")
        if self.anchor_mode == "adapter":
            anchor_file = find_adapter_file(os.environ.get("EWC_ANCHOR_ADAPTER", ""))
            if anchor_file is None:
                raise FileNotFoundError(
                    "Cannot find adapter_model.safetensors from "
                    f"EWC_ANCHOR_ADAPTER={os.environ.get('EWC_ANCHOR_ADAPTER')}"
                )
            self.anchor = load_safetensor_map(anchor_file)
        self.fisher = load_safetensor_map(fisher_path)
        logger.info(
            f"EWC enabled: mode={self.mode} anchor_mode={self.anchor_mode} "
            f"lambda={self.weight} reduction={self.reduction} "
            f"normalize={self.normalize} apply_half={self.apply_half} fisher_tensors={len(self.fisher)} "
            f"anchor_tensors={len(self.anchor)} ba_targets={','.join(self.ba_targets)}"
        )

    @staticmethod
    def is_lora_param(name: str, param: torch.nn.Parameter) -> bool:
        if not param.requires_grad:
            return False
        normalized = normalize_key(name)
        return "lora_A" in normalized or "lora_B" in normalized

    def _tensor_to(self, kind: str, key: str, ref: torch.Tensor) -> torch.Tensor:
        if kind == "anchor" and self.anchor_mode == "initial_zero":
            cache_key = ("anchor_zero", key, ref.device, ref.dtype)
            if self.cache_enabled and cache_key in self.cache:
                return self.cache[cache_key]
            tensor = torch.zeros_like(ref)
            if self.cache_enabled:
                self.cache[cache_key] = tensor
            return tensor
        source = self.anchor if kind == "anchor" else self.fisher
        cache_key = (kind, key, ref.device, ref.dtype)
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        tensor = source[key]
        dtype = ref.dtype if kind == "anchor" else torch.float32
        tensor = tensor.to(device=ref.device, dtype=dtype, non_blocking=True)
        if kind == "fisher":
            if self.normalize == "mean":
                tensor = tensor / (tensor.mean().clamp_min(1e-12))
            elif self.normalize == "max":
                tensor = tensor / (tensor.max().clamp_min(1e-12))
        if self.cache_enabled:
            self.cache[cache_key] = tensor
        return tensor

    @staticmethod
    def _active_lora_adapter(module: torch.nn.Module) -> Optional[str]:
        active_adapters = getattr(module, "active_adapters", None)
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None:
            return None
        if active_adapters:
            for adapter in active_adapters:
                if adapter in lora_a:
                    return adapter
        keys = list(lora_a.keys())
        return keys[0] if keys else None

    def _is_ba_target(self, module_key: str) -> bool:
        return any(module_key.endswith(f".{target}") or module_key.endswith(target) for target in self.ba_targets)

    def _fisher_ba_to(self, module_key: str, ref: torch.Tensor) -> torch.Tensor:
        cache_key = ("fisher_ba", module_key, ref.device, torch.float32)
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        tensor = self.fisher[module_key].to(device=ref.device, dtype=torch.float32, non_blocking=True)
        if self.normalize == "mean":
            tensor = tensor / tensor.mean().clamp_min(1e-12)
        elif self.normalize == "max":
            tensor = tensor / tensor.max().clamp_min(1e-12)
        if self.cache_enabled:
            self.cache[cache_key] = tensor
        return tensor

    def _anchor_ba_to(self, module_key: str, ref: torch.Tensor, scaling: float) -> torch.Tensor:
        cache_key = ("anchor_ba", module_key, ref.device, torch.float32)
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        if self.anchor_mode == "initial_zero":
            anchor_ba = torch.zeros_like(ref, dtype=torch.float32)
            if self.cache_enabled:
                self.cache[cache_key] = anchor_ba
            return anchor_ba
        a_key = f"{module_key}.lora_A.weight"
        b_key = f"{module_key}.lora_B.weight"
        if a_key not in self.anchor or b_key not in self.anchor:
            raise KeyError(f"missing anchor LoRA pair for {module_key}: {a_key}, {b_key}")
        a_weight = self.anchor[a_key].to(device=ref.device, dtype=torch.float32, non_blocking=True)
        b_weight = self.anchor[b_key].to(device=ref.device, dtype=torch.float32, non_blocking=True)
        anchor_ba = b_weight @ a_weight
        anchor_ba = anchor_ba * scaling
        if self.cache_enabled:
            self.cache[cache_key] = anchor_ba
        return anchor_ba

    def _loss_device(self, model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    def _lora_param_penalty(self, model: torch.nn.Module) -> tuple[torch.Tensor, int, int]:
        loss_device = self._loss_device(model)
        total: Optional[torch.Tensor] = None
        count = 0
        matched = 0
        for name, param in model.named_parameters():
            if not self.is_lora_param(name, param):
                continue
            key = normalize_key(name)
            anchor_missing = self.anchor_mode == "adapter" and key not in self.anchor
            if key not in self.fisher or anchor_missing:
                if self.warn_missing and key not in self._missing:
                    logger.warning(f"EWC missing fisher/anchor for trainable LoRA parameter: {key}")
                    self._missing.add(key)
                continue
            fisher = self._tensor_to("fisher", key, param)
            anchor = self._tensor_to("anchor", key, param)
            diff = param.float() - anchor.float()
            value = (fisher * diff.square()).sum().to(loss_device)
            total = value if total is None else total + value
            count += diff.numel()
            matched += 1
        if total is None:
            total = torch.zeros((), device=loss_device)
        return total, matched, count

    def _ba_penalty(self, model: torch.nn.Module) -> tuple[torch.Tensor, int, int]:
        loss_device = self._loss_device(model)
        total: Optional[torch.Tensor] = None
        count = 0
        matched = 0
        for module_name, module in model.named_modules():
            module_key = normalize_key(module_name)
            if not self._is_ba_target(module_key):
                continue
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            adapter = self._active_lora_adapter(module)
            if adapter is None:
                continue
            if module_key not in self.fisher:
                if self.warn_missing and module_key not in self._missing:
                    logger.warning(f"EWC missing BA fisher for LoRA module: {module_key}")
                    self._missing.add(module_key)
                continue
            lora_a = module.lora_A[adapter]
            lora_b = module.lora_B[adapter]
            scaling = float(module.scaling[adapter])
            current_ba = (lora_b.weight.float() @ lora_a.weight.float()) * scaling
            anchor_ba = self._anchor_ba_to(module_key, current_ba, scaling)
            fisher = self._fisher_ba_to(module_key, current_ba)
            diff = current_ba - anchor_ba
            value = (fisher * diff.square()).sum().to(loss_device)
            total = value if total is None else total + value
            count += diff.numel()
            matched += 1
        if total is None:
            total = torch.zeros((), device=loss_device)
        return total, matched, count

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        if not self.enabled or self.weight == 0:
            return torch.zeros((), device=next(model.parameters()).device)
        if self.mode == "ba_kv":
            total, matched, count = self._ba_penalty(model)
        else:
            total, matched, count = self._lora_param_penalty(model)
        if not self._reported:
            logger.info(f"EWC matched tensors/modules: {matched}, elements={count}")
            self._reported = True
        if self.reduction == "mean":
            total = total / max(count, 1)
        if self.apply_half:
            total = total * 0.5
        return total


class EwcSwiftSft(SwiftSft):
    def __init__(self, args=None) -> None:
        self.ewc = SwiftEWC()
        super().__init__(args)

    def _get_trainer_kwargs(self):
        kwargs = super()._get_trainer_kwargs()
        if not self.ewc.enabled:
            return kwargs

        def compute_loss_func(outputs, labels, num_items_in_batch=None, loss_scale=None, trainer=None):
            per_token_loss = per_token_loss_func(outputs, labels, enable_dft_loss=trainer.args.enable_dft_loss)
            if loss_scale is not None:
                loss_scale_flat = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                per_token_loss = per_token_loss * loss_scale_flat.to(per_token_loss.device)
            if num_items_in_batch is None:
                denom = (labels != -100).sum().clamp_min(1).to(per_token_loss.device)
            else:
                denom = torch.as_tensor(num_items_in_batch, device=per_token_loss.device).clamp_min(1)
            sft_loss = per_token_loss.sum() / denom
            ewc_loss = self.ewc.penalty(trainer.model)
            weighted = self.ewc.weight * ewc_loss
            if hasattr(trainer, "custom_metrics"):
                trainer.custom_metrics["train"]["sft_loss"].update(sft_loss.detach())
                trainer.custom_metrics["train"]["ewc_loss"].update(ewc_loss.detach())
                trainer.custom_metrics["train"]["ewc_weighted"].update(weighted.detach())
            return sft_loss + weighted

        kwargs["compute_loss_func"] = compute_loss_func
        return kwargs


def main() -> None:
    EwcSwiftSft(sys.argv[1:]).main()


if __name__ == "__main__":
    main()
