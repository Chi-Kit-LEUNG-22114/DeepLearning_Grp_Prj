from __future__ import annotations

import torch
from torch import nn

from .common import ModalityAdapters, ScalarRegressionHead, TransferModelBase, ordered_feature_groups


class Chronos2MultimodalRegressor(TransferModelBase):
    """Chronos-2 encoder with modality adapters and a scalar BMI head."""

    def __init__(
        self,
        feature_groups: dict[str, list[int]],
        model_config: dict,
        dropout: float = 0.2,
    ):
        super().__init__()
        try:
            from chronos import Chronos2Pipeline
            from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 requires chronos-forecasting>=2.2.0."
            ) from exc

        groups = ordered_feature_groups(feature_groups)
        latent_per_block = int(model_config.get("latent_channels_per_block", 2))
        self.adapters = ModalityAdapters(groups, latent_per_block, dropout=dropout)
        self.block_names = list(groups)
        self.block_slices = dict(self.adapters.channel_slices)
        self.num_output_patches = int(model_config.get("num_output_patches", 1))
        self.model_id = model_config["model_id"]

        # Keep the library's public pipeline for model loading and access its
        # differentiable core encoder for the custom multimodal regression task.
        backbone_config = model_config.get("backbone_config")
        if backbone_config:
            config = Chronos2CoreConfig.from_dict(backbone_config)
            self.pretrained = Chronos2Model(config)
            self._pipeline = None
        else:
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id, device_map="cpu"
            )
            self.pretrained = self._pipeline.model
        hidden_width = int(self.pretrained.config.d_model)
        self.regression_head = ScalarRegressionHead(
            len(groups) * hidden_width, hidden_width=192, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One compact scalar history per modality adapter channel. Chronos-2
        # treats these histories as related variates; group attention lets the
        # pretrained encoder exchange information across blocks.
        latent = self.adapters(x)  # [B, context, latent variates]
        batch_size, context_length, num_variates = latent.shape
        context = latent.transpose(1, 2).reshape(
            batch_size * num_variates, context_length
        )
        group_ids = torch.arange(
            batch_size, device=context.device, dtype=torch.long
        ).repeat_interleave(num_variates)
        encoded = self.pretrained.encode(
            context=context,
            group_ids=group_ids,
            num_output_patches=self.num_output_patches,
        )
        encoder_output, _, _, _ = encoded
        hidden = encoder_output.last_hidden_state
        # encode() appends a future query token after the optional register
        # token. The final token is the contextual representation for BMI.
        query = hidden[:, -1, :].reshape(batch_size, num_variates, -1)
        block_representations = []
        for name in self.block_names:
            start, end = self.block_slices[name]
            block_representations.append(query[:, start:end].mean(dim=1))
        fused = torch.cat(block_representations, dim=-1)
        return self.regression_head(fused)
