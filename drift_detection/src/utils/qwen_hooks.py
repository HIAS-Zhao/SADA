# -*- coding: utf-8 -*-
"""
Qwen2.5-VL Feature Extractor for drift detection experiments.

This module implements multi-stream orthogonal signal extraction:
- Stream A: Visual Perception (ViT output global pooling)
- Stream B: LLM-aligned visual state (Decoder hidden states, image tokens only)
- Stream C: Behavioral Uncertainty (52-dim logits fingerprint)
- Stream D: Enhanced Cognition (Stream B + Stream C concatenation)
- Stream E_last: Task-conditioned fusion state (last valid decoder hidden state)
- Stream E_text: Task-conditioned fusion state (text-side decoder hidden mean after last image)

Key Engineering Details:
1. Padding Defense: Use attention_mask to locate last valid token
2. Image-Text Separation: Use vision token IDs (151652, 151653) for masking
3. Uncertainty Fingerprint: Top-50 values + Max Prob + Energy Score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Token ID Constants
# ============================================================================

VISION_START_TOKEN_ID = 151652  # <|vision_start|>
VISION_END_TOKEN_ID = 151653    # <|vision_end|>


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ExtractedFeatures:
    """Extracted multi-stream features."""
    A: torch.Tensor  # Stream A: Visual Perception (ViT global pooling)
    B: torch.Tensor  # Stream B: LLM-aligned visual state (Decoder hidden, image tokens)
    C: torch.Tensor  # Stream C: Behavioral Uncertainty (52-dim logits fingerprint)
    D: torch.Tensor  # Stream D: Enhanced Cognition (B + C concatenation)
    E_last: torch.Tensor  # Stream E_last: Last valid token hidden state
    E_text: torch.Tensor  # Stream E_text: Text-side hidden-state mean after last image
    predicted_labels: torch.Tensor  # Predicted class labels

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary for saving."""
        return {
            'A': self.A.cpu(),
            'B': self.B.cpu(),
            'C': self.C.cpu(),
            'D': self.D.cpu(),
            'E_last': self.E_last.cpu(),
            'E_text': self.E_text.cpu(),
            'predicted_labels': self.predicted_labels.cpu(),
        }


# ============================================================================
# Feature Extractor
# ============================================================================

class FeatureExtractor:
    """
    Feature extractor with hooks for Qwen2.5-VL model.

    Implements defensive engineering to handle:
    - Dynamic resolution (variable image token lengths)
    - Padding tokens in batch inference
    - Precise image-text token separation
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int = 1536,
        device: str = "cuda",
        decoder_layer_offset: int = -1,
    ):
        """
        Initialize feature extractor.

        Args:
            model: Qwen2.5-VL model instance
            hidden_dim: Hidden dimension size (default: 1536 for 3B model)
            device: Device to run on
            decoder_layer_offset: Relative decoder layer index to hook, -1 means last layer
        """
        self.model = model
        self.hidden_dim = hidden_dim
        self.device = device
        self.decoder_layer_offset = decoder_layer_offset
        self.decoder_layer_index: Optional[int] = None

        # Storage for hooked features
        self.visual_output = None      # ViT output
        self.decoder_hidden = None     # Decoder selected layer hidden states
        self.lm_head_output = None     # LM head logits

        # Hook handles
        self.hook_handles = []

    # ========================================================================
    # Hook Functions
    # ========================================================================

    def _hook_visual_output(self, module, input, output):
        """
        Hook for visual encoder output.

        Stores raw output without pooling. Pooling will be done later
        with proper masking to handle multiple images.
        """
        if isinstance(output, tuple):
            output = output[0]

        # Store raw output (batch, seq_len, dim) or (seq_len, dim)
        self.visual_output = output

    def _hook_decoder_hidden(self, module, input, output):
        """
        Hook for decoder last layer hidden states.

        Captures: (batch, seq_len, hidden_dim)
        """
        if isinstance(output, tuple):
            self.decoder_hidden = output[0]
        else:
            self.decoder_hidden = output

    def _hook_lm_head(self, module, input, output):
        """
        Hook for LM head output (logits).

        Captures: (batch, seq_len, vocab_size)
        """
        if isinstance(output, tuple):
            self.lm_head_output = output[0]
        else:
            self.lm_head_output = output

    # ========================================================================
    # Hook Registration
    # ========================================================================

    def register_hooks(self):
        """Register forward hooks on model components."""
        try:
            # Hook 1: Visual encoder output
            visual_module = None
            if hasattr(self.model, 'visual'):
                visual_module = self.model.visual
            elif hasattr(self.model, 'vision_model'):
                visual_module = self.model.vision_model

            if visual_module:
                handle = visual_module.register_forward_hook(self._hook_visual_output)
                self.hook_handles.append(handle)
                logger.info("✓ Registered hook on visual encoder")
            else:
                logger.warning("⚠ Visual encoder not found")

            # Hook 2: Decoder selected layer
            # Qwen2.5-VL structure: model.language_model.layers
            decoder_layers = None
            if hasattr(self.model, 'language_model'):
                if hasattr(self.model.language_model, 'layers'):
                    decoder_layers = self.model.language_model.layers

            if decoder_layers is not None and len(decoder_layers) > 0:
                layer_count = len(decoder_layers)
                target_index = self.decoder_layer_offset
                if target_index < 0:
                    target_index = layer_count + target_index
                if target_index < 0 or target_index >= layer_count:
                    raise ValueError(
                        f"decoder_layer_offset={self.decoder_layer_offset} is out of range for {layer_count} decoder layers"
                    )

                self.decoder_layer_index = target_index
                selected_layer = decoder_layers[target_index]
                handle = selected_layer.register_forward_hook(self._hook_decoder_hidden)
                self.hook_handles.append(handle)
                logger.info(
                    "✓ Registered hook on decoder layer index %s (offset=%s)",
                    target_index,
                    self.decoder_layer_offset,
                )
            else:
                logger.warning("⚠ Decoder layers not found")

            # Hook 3: LM head
            if hasattr(self.model, 'lm_head'):
                handle = self.model.lm_head.register_forward_hook(self._hook_lm_head)
                self.hook_handles.append(handle)
                logger.info("✓ Registered hook on LM head")
            else:
                logger.warning("⚠ LM head not found")

        except Exception as e:
            logger.error(f"Error registering hooks: {e}")
            raise

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.info("✓ Removed all hooks")

    def clear_cache(self):
        """Clear cached hook outputs."""
        self.visual_output = None
        self.decoder_hidden = None
        self.lm_head_output = None

    # ========================================================================
    # Core Feature Extraction Logic
    # ========================================================================

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> ExtractedFeatures:
        """
        Extract multi-stream features from model forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            pixel_values: Image pixel values (optional)
            image_grid_thw: Image grid dimensions (optional)

        Returns:
            ExtractedFeatures containing streams A, B, C, D, E_last, E_text
        """
        self.clear_cache()

        # Forward pass (hooks will capture intermediate outputs)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        # Extract logits (with padding defense)
        logits = self.lm_head_output if self.lm_head_output is not None else outputs.logits

        # ====================================================================
        # Stream A: Visual Perception (Per-Image Pooling)
        # ====================================================================
        stream_A = self._extract_stream_A(attention_mask, image_grid_thw)

        # ====================================================================
        # Stream B: LLM-Aligned Visual State (Decoder Hidden, Image Tokens Only)
        # ====================================================================
        stream_B = self._extract_stream_B(input_ids, attention_mask)

        # ====================================================================
        # Stream C: Behavioral Uncertainty (52-dim Logits Fingerprint)
        # ====================================================================
        stream_C = self._extract_stream_C(logits, attention_mask)

        # ====================================================================
        # Stream D: Enhanced Cognition (B + C Concatenation)
        # ====================================================================
        stream_D = torch.cat([stream_B, stream_C], dim=-1)

        # ====================================================================
        # Stream E: Task-Conditioned Fusion States
        # ====================================================================
        stream_E_last = self._extract_stream_E_last(input_ids, attention_mask)
        stream_E_text = self._extract_stream_E_text(input_ids, attention_mask)

        # ====================================================================
        # Predicted Labels
        # ====================================================================
        predicted_labels = self._get_predicted_labels(logits, attention_mask)

        return ExtractedFeatures(
            A=stream_A,
            B=stream_B,
            C=stream_C,
            D=stream_D,
            E_last=stream_E_last,
            E_text=stream_E_text,
            predicted_labels=predicted_labels,
        )

    # ========================================================================
    # Stream Extraction Helpers
    # ========================================================================

    def _extract_stream_A(
        self,
        attention_mask: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract Stream A: Visual Perception (Per-Image Pooling to avoid Resolution Bias).

        Logic:
        1. Get raw visual encoder output
        2. Use image_grid_thw to split patches by image
        3. Pool each image independently (equal weight per image)
        4. Average across all images for the sample

        Args:
            attention_mask: (batch, seq_len)
            image_grid_thw: (num_images, 3) where each row is (temporal, height, width)

        Returns:
            stream_A: (batch, hidden_dim)
        """
        if self.visual_output is None:
            batch_size = attention_mask.shape[0]
            logger.warning("⚠ Visual output not captured, using zeros for Stream A")
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)

        visual_out = self.visual_output

        # Handle different output formats
        if visual_out.dim() == 3:
            # (batch, seq_len, dim) - standard format
            batch_size, seq_len, hidden_dim = visual_out.shape

            # For ViT output, all tokens are valid (no padding in visual encoder)
            # Apply mean pooling across all visual tokens (all images)
            stream_A = visual_out.mean(dim=1)  # (batch, hidden_dim)

        elif visual_out.dim() == 2:
            # (seq_len, dim) - no batch dimension
            if image_grid_thw is not None:
                patch_lengths = image_grid_thw.prod(dim=1)  # (num_images,)
                total_patches = patch_lengths.sum().item()
                actual_patches = visual_out.shape[0]

                # 计算 ViT Patch 与 LLM Token 的比例 (Qwen2.5-VL 通常是 4)
                # total_patches 是 image_grid_thw 计算的，actual_patches 是实际的 visual_out 长度
                # 通常 total_patches > actual_patches，因为 image_grid_thw 是合并前的
                if actual_patches > 0 and total_patches % actual_patches == 0:
                    ratio = total_patches // actual_patches

                    # 按比例缩小每张图的长度，精确匹配 visual_out 的边界
                    actual_lengths = [p.item() // ratio for p in patch_lengths]

                    # 按图片精准切分
                    image_features = torch.split(visual_out, actual_lengths, dim=0)

                    # 真正的多图平权：先单图平均，再整体平均
                    pooled_images = [img.mean(dim=0) for img in image_features]
                    stream_A = torch.stack(pooled_images).mean(dim=0, keepdim=True)  # (1, hidden_dim)

                else:
                    # 只有在遇到极度异常的非整数倍 mismatch 时，才触发 Fallback
                    logger.warning(f"⚠ Unhandled patch ratio: total={total_patches}, actual={actual_patches}. Using fallback pooling.")
                    stream_A = visual_out.mean(dim=0, keepdim=True)
            else:
                stream_A = visual_out.mean(dim=0, keepdim=True)

        else:
            # Fallback: flatten and mean pool
            batch_size = attention_mask.shape[0]
            stream_A = visual_out.view(batch_size, -1, visual_out.shape[-1]).mean(dim=1)

        return stream_A

    def _extract_stream_B(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract Stream B: Multimodal Cognition with Masked Mean Pooling.

        Logic:
        1. Locate ALL vision token pairs using VISION_START_TOKEN_ID and VISION_END_TOKEN_ID
        2. Create binary mask for all image tokens (supports multiple images)
        3. Extract decoder hidden states for all image tokens
        4. Apply masked mean pooling: (features * mask).sum() / mask.sum()

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            stream_B: (batch, hidden_dim)
        """
        if self.decoder_hidden is None:
            batch_size = input_ids.shape[0]
            logger.warning("⚠ Decoder hidden not captured, using zeros for Stream B")
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)

        batch_size, seq_len, hidden_dim = self.decoder_hidden.shape
        stream_B_list = []

        for i in range(batch_size):
            # Find ALL vision token ranges
            ids = input_ids[i]
            start_positions = (ids == VISION_START_TOKEN_ID).nonzero(as_tuple=True)[0]
            end_positions = (ids == VISION_END_TOKEN_ID).nonzero(as_tuple=True)[0]

            # Build binary mask for image tokens
            image_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)

            n_pairs = min(len(start_positions), len(end_positions))
            for pair_idx in range(n_pairs):
                start_idx = start_positions[pair_idx].item()
                end_idx = end_positions[pair_idx].item()

                # Mark image tokens (exclusive of special tokens)
                if start_idx + 1 < end_idx:
                    image_mask[start_idx+1:end_idx] = True

            # Apply masked mean pooling
            if image_mask.any():
                # Masked mean: (features * mask).sum() / mask.sum()
                masked_hidden = self.decoder_hidden[i] * image_mask.unsqueeze(-1).float()
                stream_B_i = masked_hidden.sum(dim=0) / image_mask.sum().float()
            else:
                # No image tokens found
                stream_B_i = torch.zeros(hidden_dim, device=self.device)

            stream_B_list.append(stream_B_i)

        stream_B = torch.stack(stream_B_list, dim=0)
        return stream_B

    def _extract_stream_C(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract Stream C: Behavioral Uncertainty (52-dim fingerprint).

        Logic:
        1. Use attention_mask to locate last valid token (padding defense)
        2. Extract logits at last valid position
        3. Compute 52-dim fingerprint:
           - Top-50 logit values (sorted descending)
           - Max probability (after softmax)
           - Energy score (logsumexp)

        Args:
            logits: (batch, seq_len, vocab_size)
            attention_mask: (batch, seq_len)

        Returns:
            stream_C: (batch, 52)
        """
        batch_size = logits.shape[0]
        fingerprints = []

        for i in range(batch_size):
            # Find last valid token position (padding defense)
            valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_valid_idx = valid_positions[-1].item()
            else:
                last_valid_idx = 0

            # Extract logits at last valid position
            logits_i = logits[i, last_valid_idx, :]  # (vocab_size,)

            # Compute 52-dim fingerprint
            fingerprint = self._compute_uncertainty_fingerprint(logits_i)
            fingerprints.append(fingerprint)

        stream_C = torch.stack(fingerprints, dim=0)
        return stream_C

    def _extract_stream_E_last(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract Stream E_last: final valid decoder hidden state.

        This token is after the image tokens and user text in the Qwen chat
        template, so causal attention lets it see both visual and textual
        context. It is the sharper but more template-sensitive fusion signal.
        """
        if self.decoder_hidden is None:
            batch_size = input_ids.shape[0]
            logger.warning("Decoder hidden not captured, using zeros for Stream E_last")
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)

        rows = []
        for i in range(input_ids.shape[0]):
            valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
            last_idx = valid_positions[-1].item() if len(valid_positions) > 0 else 0
            rows.append(self.decoder_hidden[i, last_idx])

        return torch.stack(rows, dim=0)

    def _extract_stream_E_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract Stream E_text: mean-pooled text-side decoder hidden states.

        Tokens after the last visual segment can attend to all earlier image
        tokens and preceding task text. Mean pooling over this region gives a
        stable task-conditioned image-text fusion representation.
        """
        if self.decoder_hidden is None:
            batch_size = input_ids.shape[0]
            logger.warning("Decoder hidden not captured, using zeros for Stream E_text")
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)

        batch_size, seq_len, hidden_dim = self.decoder_hidden.shape
        rows = []

        for i in range(batch_size):
            ids = input_ids[i]
            valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                rows.append(torch.zeros(hidden_dim, device=self.device))
                continue

            last_valid = valid_positions[-1].item()
            end_positions = (ids == VISION_END_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                start = min(end_positions[-1].item() + 1, last_valid)
            else:
                start = valid_positions[0].item()

            token_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            token_mask[start:last_valid + 1] = True
            token_mask &= attention_mask[i].bool()

            if token_mask.any():
                rows.append(self.decoder_hidden[i][token_mask].mean(dim=0))
            else:
                rows.append(self.decoder_hidden[i, last_valid])

        return torch.stack(rows, dim=0)

    def _compute_uncertainty_fingerprint(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute 52-dimensional uncertainty fingerprint.

        Components:
        1. Top-50 logit values (shape information)
        2. Max probability (confidence)
        3. Energy score (robustness)

        Args:
            logits: (vocab_size,)

        Returns:
            fingerprint: (52,)
        """
        # 1. Top-50 values (sorted descending)
        top50_values, _ = torch.topk(logits, k=50, largest=True, sorted=True)

        # 2. Max probability
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max()

        # 3. Energy score (logsumexp)
        energy = torch.logsumexp(logits, dim=-1)

        # Concatenate into 52-dim vector
        fingerprint = torch.cat([
            top50_values,           # 50 dims
            max_prob.unsqueeze(0),  # 1 dim
            energy.unsqueeze(0),    # 1 dim
        ])

        return fingerprint

    def _get_predicted_labels(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predicted labels from logits.

        Args:
            logits: (batch, seq_len, vocab_size)
            attention_mask: (batch, seq_len)

        Returns:
            predicted_labels: (batch,)
        """
        batch_size = logits.shape[0]
        labels = []

        for i in range(batch_size):
            # Find last valid token position
            valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_valid_idx = valid_positions[-1].item()
            else:
                last_valid_idx = 0

            # Get predicted label
            logits_i = logits[i, last_valid_idx, :]
            label = logits_i.argmax(dim=-1)
            labels.append(label)

        predicted_labels = torch.stack(labels, dim=0)
        return predicted_labels


# ============================================================================
# Factory Function
# ============================================================================

def create_extractor(
    model: nn.Module,
    device: str = "cuda",
    decoder_layer_offset: int = -1,
) -> FeatureExtractor:
    """
    Create and initialize a FeatureExtractor.

    Args:
        model: Qwen2.5-VL model instance
        device: Device to run on
        decoder_layer_offset: Relative decoder layer index to hook, -1 means last layer

    Returns:
        Initialized FeatureExtractor with hooks registered
    """
    # Detect hidden dimension
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = 1536  # Default for Qwen2.5-VL-3B

    extractor = FeatureExtractor(
        model,
        hidden_dim=hidden_dim,
        device=device,
        decoder_layer_offset=decoder_layer_offset,
    )
    extractor.register_hooks()

    logger.info(
        "✓ Created FeatureExtractor (hidden_dim=%s, decoder_layer_offset=%s, decoder_layer_index=%s)",
        hidden_dim,
        decoder_layer_offset,
        extractor.decoder_layer_index,
    )
    return extractor
