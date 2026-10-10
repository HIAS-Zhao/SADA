# -*- coding: utf-8 -*-
"""
Qwen2.5-VL sequence-level feature extractor.

This module is independent from existing qwen_hooks.py and is designed to keep
sequence information instead of collapsing all tokens into one vector.

Outputs are per-sample 2D tensors:
- Stream A: vision-token features, shape (K_a, D)
- Stream B: window-pooled multimodal features on image-token spans, shape (K_b, D)
- Stream C: window-pooled multimodal features on valid tokens, shape (K_c, D)
- Stream D: concatenated window-pooled features, shape (K_d, 2D)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

VISION_START_TOKEN_ID = 151652  # <|vision_start|>
VISION_END_TOKEN_ID = 151653    # <|vision_end|>


@dataclass
class SequenceFeatures:
    """Per-sample sequence-level feature package."""
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    predicted_labels: torch.Tensor


class SequenceFeatureExtractor:
    """
    Hook-based sequence feature extractor.

    Notes:
    - No global mean pooling over full sequence.
    - Stream A keeps vision token features as-is.
    - Stream B/C use local window mean pooling.
    """

    def __init__(self, model: nn.Module, window_size: int = 100, device: str = "cuda"):
        self.model = model
        self.window_size = int(window_size)
        self.device = device

        self.visual_output = None
        self.decoder_hidden = None
        self.lm_head_output = None
        self.hook_handles: List = []

    def _hook_visual_output(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        self.visual_output = output

    def _hook_decoder_hidden(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        self.decoder_hidden = output

    def _hook_lm_head(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        self.lm_head_output = output

    def register_hooks(self):
        visual_module = None
        if hasattr(self.model, "visual"):
            visual_module = self.model.visual
        elif hasattr(self.model, "vision_model"):
            visual_module = self.model.vision_model

        if visual_module is not None:
            self.hook_handles.append(visual_module.register_forward_hook(self._hook_visual_output))

        decoder_layers = None
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            decoder_layers = self.model.language_model.layers

        if decoder_layers is not None and len(decoder_layers) > 0:
            self.hook_handles.append(decoder_layers[-1].register_forward_hook(self._hook_decoder_hidden))

        if hasattr(self.model, "lm_head"):
            self.hook_handles.append(self.model.lm_head.register_forward_hook(self._hook_lm_head))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def clear_cache(self):
        self.visual_output = None
        self.decoder_hidden = None
        self.lm_head_output = None

    def _window_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Local mean pooling over sequence axis with fixed window size."""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor for window pooling, got {x.shape}")
        n = x.size(0)
        if n == 0:
            return x

        chunks: List[torch.Tensor] = []
        for i in range(0, n, self.window_size):
            chunk = x[i:i + self.window_size]
            if chunk.numel() == 0:
                continue
            chunks.append(chunk.mean(dim=0, keepdim=True))

        return torch.cat(chunks, dim=0) if chunks else x[:0]

    def _collect_image_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Create bool mask for all image tokens between vision start/end tokens."""
        seq_len = ids.size(0)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)

        starts = (ids == VISION_START_TOKEN_ID).nonzero(as_tuple=True)[0]
        ends = (ids == VISION_END_TOKEN_ID).nonzero(as_tuple=True)[0]
        n = min(len(starts), len(ends))

        for i in range(n):
            s = starts[i].item()
            e = ends[i].item()
            if s + 1 < e:
                mask[s + 1:e] = True

        return mask

    def _last_valid_index(self, attn: torch.Tensor) -> int:
        valid = attn.nonzero(as_tuple=True)[0]
        return valid[-1].item() if len(valid) > 0 else 0

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> List[SequenceFeatures]:
        """
        Return one SequenceFeatures object per sample in current batch.
        """
        self.clear_cache()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        logits = self.lm_head_output if self.lm_head_output is not None else outputs.logits
        if self.decoder_hidden is None:
            raise RuntimeError("decoder hidden is not captured by hooks")

        batch_size = input_ids.size(0)
        result: List[SequenceFeatures] = []

        for i in range(batch_size):
            ids_i = input_ids[i]
            attn_i = attention_mask[i]
            dec_i = self.decoder_hidden[i]  # (seq, D)

            # Stream A: keep vision token sequence, no global pooling.
            image_mask = self._collect_image_mask(ids_i)
            A_seq = dec_i[image_mask]
            if A_seq.numel() == 0:
                # Fallback: keep all valid tokens if image mask is empty
                A_seq = dec_i[attn_i.bool()]

            # Stream B: local window pooling on image-token sequence.
            B_seq = self._window_pool(A_seq)

            # Stream C: local window pooling on all valid tokens (sequence context).
            C_raw = dec_i[attn_i.bool()]
            C_seq = self._window_pool(C_raw)

            # Stream D: concatenate pooled B and C, align by min window count.
            k = min(B_seq.size(0), C_seq.size(0))
            if k == 0:
                D_seq = torch.zeros((0, dec_i.size(-1) * 2), device=dec_i.device, dtype=dec_i.dtype)
            else:
                D_seq = torch.cat([B_seq[:k], C_seq[:k]], dim=-1)

            last_idx = self._last_valid_index(attn_i)
            pred = logits[i, last_idx].argmax(dim=-1).view(1)

            result.append(
                SequenceFeatures(
                    A=A_seq,
                    B=B_seq,
                    C=C_seq,
                    D=D_seq,
                    predicted_labels=pred,
                )
            )

        return result


def create_sequence_extractor(model: nn.Module, window_size: int = 100, device: str = "cuda") -> SequenceFeatureExtractor:
    extractor = SequenceFeatureExtractor(model=model, window_size=window_size, device=device)
    extractor.register_hooks()
    logger.info("Created SequenceFeatureExtractor (window_size=%d)", window_size)
    return extractor
