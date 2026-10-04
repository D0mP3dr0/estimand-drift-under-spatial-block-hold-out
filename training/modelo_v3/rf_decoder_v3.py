#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Affine RF decoder used by the third-campaign GNN and MLP trainers.

The decoder replaces the softplus/sigmoid/clamp output stage of the production
`PhysicsConstrainedDecoder` by a fixed affine map, so that the training loss
never sees a saturated output. It subclasses the base `RFDecoder` of the
original model package (imported from `02_models`, not modified) and reuses its
shared MLP trunk `self.layers`; the unused heads of the base class are kept, so
the parameter count matches the original decoder.

Output columns and units (the target order of the production loss, which applies
a Huber loss column by column with no internal rescaling):
    0 path_loss_total (dB), 1 path_loss_vegetation (dB),
    2 path_loss_terrain (dB), 3 rssi (dBm), 4 coverage_prob (0-1).
Channels 0-3 are `raw * scale + offset`, centred on the production clamp
interval with a span of half its width; channel 4 is `sigmoid(raw)`, the same
sigmoid the base `RFDecoder` applies, because its target is a probability.
The production clamps are applied only in `fisico()` (inference and reporting).

Usage: imported by `train_gnn_v3.py` and `train_mlp_v3.py`. Running the module
directly executes a CPU smoke test (seed 0) that checks the output shape, the
absence of NaN, and the clamp bounds of `fisico()`.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

_GNN_RF_V2_02_MODELS = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/02_models")
if str(_GNN_RF_V2_02_MODELS) not in sys.path:
    sys.path.insert(0, str(_GNN_RF_V2_02_MODELS))

from rf_decoder import RFDecoder  # noqa: E402 (late import, after the sys.path insertion above)


# Fixed (not learned) (scale, offset) per channel: centre and half-width of the
# production clamp interval.
_ESCALA_OFFSET_PL = (
    (100.0, 100.0),   # channel 0, path_loss_total, clamp [0, 200] dB
    (25.0, 25.0),     # channel 1, path_loss_vegetation, clamp [0, 50] dB
    (15.0, 15.0),     # channel 2, path_loss_terrain, clamp [0, 30] dB
)
_ESCALA_RSSI, _OFFSET_RSSI = 75.0, -75.0   # channel 3, rssi, clamp [-150, 0] dBm

CLAMP_PL = ((0.0, 200.0), (0.0, 50.0), (0.0, 30.0))
CLAMP_RSSI = (-150.0, 0.0)
CLAMP_COVERAGE = (0.0, 1.0)


class AffineDecoderV3(RFDecoder):
    """RF decoder with an affine output on channels 0-3 and a sigmoid on channel 4.

    Inherits the trunk `self.layers` of `RFDecoder`; the base-class heads are
    kept but never used in `forward`, which preserves the parameter count.
    """

    def __init__(self, *args, **kwargs):
        # The min/max keyword arguments are accepted for signature compatibility
        # with PhysicsConstrainedDecoder, which this class replaces at run time.
        min_path_loss = kwargs.pop("min_path_loss", CLAMP_PL[0][0])
        max_path_loss = kwargs.pop("max_path_loss", CLAMP_PL[0][1])
        min_rssi = kwargs.pop("min_rssi", CLAMP_RSSI[0])
        max_rssi = kwargs.pop("max_rssi", CLAMP_RSSI[1])
        super().__init__(*args, **kwargs)
        self.min_path_loss = min_path_loss
        self.max_path_loss = max_path_loss
        self.min_rssi = min_rssi
        self.max_rssi = max_rssi

    def forward(self, terrain_embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        """Training-time output in target units; calls the trunk directly so the sigmoid is applied once."""
        raw = self.layers(terrain_embeddings)  # [N, 5] pre-activation

        out = torch.empty_like(raw)
        for canal, (escala, offset) in enumerate(_ESCALA_OFFSET_PL):
            out[:, canal] = raw[:, canal] * escala + offset
        out[:, 3] = raw[:, 3] * _ESCALA_RSSI + _OFFSET_RSSI
        out[:, 4] = torch.sigmoid(raw[:, 4])

        return out

    def fisico(self, pred: torch.Tensor) -> torch.Tensor:
        """Clamp a prediction to the production ranges; for inference and reporting only, never in the loss."""
        out = pred.clone()
        out[:, 0] = torch.clamp(out[:, 0], *CLAMP_PL[0])
        out[:, 1] = torch.clamp(out[:, 1], *CLAMP_PL[1])
        out[:, 2] = torch.clamp(out[:, 2], *CLAMP_PL[2])
        out[:, 3] = torch.clamp(out[:, 3], *CLAMP_RSSI)
        out[:, 4] = torch.clamp(out[:, 4], *CLAMP_COVERAGE)
        return out


def sha256_do_arquivo(caminho: Path, bloco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            chunk = f.read(bloco)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    # CPU smoke test; the embeddings are scaled up on purpose to probe saturation.
    torch.manual_seed(0)
    dec = AffineDecoderV3(in_channels=64, hidden_channels=32, out_channels=5)
    emb = torch.randn(2000, 64) * 8.0
    pred = dec(emb)
    assert pred.shape == (2000, 5), pred.shape
    assert not torch.isnan(pred).any(), "NaN na saida do decodificador V3"

    # Fraction of |pre-activation| > 4 on channels 0-3, where a softplus or
    # sigmoid output would saturate; the affine map has no plateau there.
    raw = dec.layers(emb)
    sat_0a3 = float((raw[:, :4].abs() > 4.0).float().mean())
    print(f"fracao |pre-ativacao|>4 nos canais 0-3 (nao usada para saturar mais): {sat_0a3:.4f}")

    fis = dec.fisico(pred)
    assert fis[:, 0].min() >= 0.0 and fis[:, 0].max() <= 200.0
    assert fis[:, 1].min() >= 0.0 and fis[:, 1].max() <= 50.0
    assert fis[:, 2].min() >= 0.0 and fis[:, 2].max() <= 30.0
    assert fis[:, 3].min() >= -150.0 and fis[:, 3].max() <= 0.0
    assert fis[:, 4].min() >= 0.0 and fis[:, 4].max() <= 1.0

    print("OK: AffineDecoderV3 smoke de CPU passou.", pred.shape,
          "| sha256(este arquivo)=", sha256_do_arquivo(Path(__file__)))
