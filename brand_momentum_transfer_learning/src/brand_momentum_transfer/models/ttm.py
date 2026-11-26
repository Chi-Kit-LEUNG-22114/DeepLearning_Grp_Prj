from __future__ import annotations

import torch
from torch import nn

from .common import ModalityAdapters, ScalarRegressionHead, TransferModelBase, ordered_feature_groups


class TTMMultimodalRegressor(TransferModelBase):
    """TTM r2.1 temporal backbone plus a multimodal BMI regression head."""

    def __init__(
        self,
        feature_groups: dict[str, list[int]],
        model_config: dict,
        dropout: float = 0.2,
    ):
        super().__init__()
        try:
            from tsfm_public.models.tinytimemixer import (
                TinyTimeMixerConfig,
                TinyTimeMixerForPrediction,
            )
        except ImportError as exc:
            raise ImportError(
                "TTM requires granite-tsfm. Install the project requirements first."
            ) from exc

        groups = ordered_feature_groups(feature_groups)
        latent_per_block = int(model_config.get("latent_channels_per_block", 1))
        self.adapters = ModalityAdapters(groups, latent_per_block, dropout=dropout)
        self.block_names = list(groups)
        self.block_slices = dict(self.adapters.channel_slices)
        self.context_length = int(model_config.get("context_length", 52))
        self.prediction_length = int(model_config.get("prediction_length", 16))
        self.frequency_token = int(model_config.get("frequency_token", 9))
        self.model_id = model_config["model_id"]
        self.revision = model_config["revision"]

        # The r2.1 52-16 checkpoint is the matching pretrained weekly-context
        # variant. TTM's channel-independent pretrained mixer can be adapted to
        # the compact channels emitted by the modality adapters.
        backbone_config = model_config.get("backbone_config")
        if backbone_config:
            # Inference can reconstruct the architecture from the saved config
            # and load all weights from model.safetensors without a Hub request.
            config = TinyTimeMixerConfig.from_dict(backbone_config)
            self.pretrained = TinyTimeMixerForPrediction(config)
        else:
            self.pretrained = TinyTimeMixerForPrediction.from_pretrained(
                self.model_id,
                revision=self.revision,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                num_input_channels=self.adapters.output_channels,
                ignore_mismatched_sizes=True,
            )
        self.pretrained.config.num_input_channels = self.adapters.output_channels
        self.pretrained.config.prediction_length = self.prediction_length
        self.pretrained.config.context_length = self.context_length

        # Backbone hidden states are [batch, channels, patches, hidden].
        # `backbone_hidden_state` is the encoder representation, i.e. d_model
        # even when the checkpoint has a separate decoder width.
        hidden_width = int(self.pretrained.config.d_model)
        representation_width = (
            self.adapters.output_channels * hidden_width
            + self.adapters.output_channels * self.prediction_length
        )
        self.regression_head = ScalarRegressionHead(
            representation_width, hidden_width=128, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.adapters(x)
        freq_token = torch.full(
            (latent.shape[0],),
            self.frequency_token,
            dtype=torch.long,
            device=latent.device,
        )
        output = self.pretrained(
            past_values=latent,
            freq_token=freq_token,
            return_dict=True,
        )
        forecast = output.prediction_outputs
        backbone = output.backbone_hidden_state
        if forecast is None or backbone is None:
            raise RuntimeError(
                "TTM did not return prediction_outputs and backbone_hidden_state; "
                "check the installed granite-tsfm version."
            )
        # Average over TTM patches, retaining a separate latent channel for
        # each modality block before the scalar BMI head.
        hidden_summary = backbone.mean(dim=2).flatten(start_dim=1)
        forecast_summary = forecast.flatten(start_dim=1)
        return self.regression_head(
            torch.cat([hidden_summary, forecast_summary], dim=-1)
        )
