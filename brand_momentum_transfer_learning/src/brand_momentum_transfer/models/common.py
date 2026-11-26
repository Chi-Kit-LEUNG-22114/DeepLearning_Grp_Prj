from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class ModalityAdapters(nn.Module):
    """Independent learned projections for the named weekly feature blocks."""

    def __init__(
        self,
        feature_groups: dict[str, list[int]],
        latent_channels_per_block: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not feature_groups:
            raise ValueError("At least one feature group is required.")
        self.group_names = list(feature_groups)
        self.group_indices = {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in feature_groups.items()
        }
        self.adapters = nn.ModuleDict()
        self.channel_slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name, indices in feature_groups.items():
            if not indices:
                continue
            self.adapters[name] = nn.Sequential(
                nn.Linear(len(indices), latent_channels_per_block),
                nn.GELU(),
                nn.LayerNorm(latent_channels_per_block),
                nn.Dropout(dropout),
            )
            self.channel_slices[name] = (
                cursor,
                cursor + latent_channels_per_block,
            )
            cursor += latent_channels_per_block
        self.output_channels = cursor
        if self.output_channels == 0:
            raise ValueError("No non-empty feature groups were supplied.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, context, feature]
        outputs = []
        for name in self.group_names:
            if name not in self.adapters:
                continue
            indices = self.group_indices[name].to(x.device)
            block = x.index_select(dim=-1, index=indices)
            outputs.append(self.adapters[name](block))
        return torch.cat(outputs, dim=-1)


class ScalarRegressionHead(nn.Module):
    def __init__(self, input_width: int, hidden_width: int = 128, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width // 2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class TransferModelBase(nn.Module):
    def set_pretrained_trainable(self, trainable: bool) -> None:
        for parameter in self.pretrained.parameters():
            parameter.requires_grad = trainable

    def pretrained_parameters(self):
        return self.pretrained.parameters()

    def head_parameters(self):
        pretrained_ids = {id(parameter) for parameter in self.pretrained.parameters()}
        return (
            parameter
            for parameter in self.parameters()
            if id(parameter) not in pretrained_ids
        )


def ordered_feature_groups(groups: dict[str, list[int]]) -> dict[str, list[int]]:
    # Stable ordering makes adapter channels and checkpoints reproducible.
    preferred = (
        "comments_bge",
        "images_clip",
        "titles_bge",
        "titles_clip",
        "numeric_context",
        "availability_quality",
    )
    ordered = OrderedDict()
    for name in preferred:
        if groups.get(name):
            ordered[name] = list(groups[name])
    for name in sorted(set(groups) - set(ordered)):
        if groups[name]:
            ordered[name] = list(groups[name])
    return dict(ordered)
