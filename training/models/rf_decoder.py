"""Decoders that map terrain-node embeddings to radio-propagation predictions.

Output columns (5 per terrain node):
    0 path_loss_total (dB), 1 path_loss_vegetation (dB),
    2 path_loss_terrain (dB), 3 rssi (dBm), 4 coverage_prob (0 to 1).

``RFDecoder`` is a three-layer GELU MLP [N, hidden_dim] -> [N, 5] with a
sigmoid on the coverage column. ``PhysicsConstrainedDecoder`` bounds the
outputs: softplus then clipping of the three path-loss columns (total to
[min_path_loss, max_path_loss], vegetation to [0, 50] dB, terrain to
[0, 30] dB) and a scaled sigmoid mapping RSSI into [min_rssi, max_rssi].

Imported by ``gnn_rf_model.py``, by the frozen graph-free (MLP) trainer and
by the third-campaign code; it reads no files and writes none. It expects
the original module layout, with the ``config`` package in the parent
directory of this file.

Usage: run the file directly for a smoke test on random embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


class RFDecoder(nn.Module):
    """MLP decoder from terrain embeddings [N, in_channels] to predictions [N, out_channels]."""
    
    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 out_channels: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        
        self.out_channels = out_channels
        
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Separate heads, used only when forward(..., use_separate_heads=True).
        self.path_loss_head = nn.Linear(out_channels, 3)
        self.rssi_head = nn.Linear(out_channels, 1)
        self.coverage_head = nn.Linear(out_channels, 1)
    
    def forward(self, 
                terrain_embeddings: torch.Tensor,
                use_separate_heads: bool = False
                ) -> torch.Tensor:
        """Return predictions [N, 5]; coverage (column 4) passes through a sigmoid.

        With ``use_separate_heads`` the shared MLP output feeds three linear heads
        (path losses [N, 3], RSSI [N, 1], coverage [N, 1]).
        """
        x = self.layers(terrain_embeddings)
        
        if use_separate_heads:
            path_loss = self.path_loss_head(x)
            rssi = self.rssi_head(x)
            coverage = torch.sigmoid(self.coverage_head(x))
            
            predictions = torch.cat([path_loss, rssi, coverage], dim=1)
        else:
            predictions = x
            predictions = torch.cat([
                predictions[:, :4],
                torch.sigmoid(predictions[:, 4:5])
            ], dim=1)
        
        return predictions
    
    def decode_predictions(self, 
                          predictions: torch.Tensor
                          ) -> Dict[str, torch.Tensor]:
        """Split predictions [N, 5] into a dict of five named [N] tensors."""
        return {
            'path_loss_total': predictions[:, 0],
            'path_loss_vegetation': predictions[:, 1],
            'path_loss_terrain': predictions[:, 2],
            'rssi': predictions[:, 3],
            'coverage_prob': predictions[:, 4]
        }


class PhysicsConstrainedDecoder(RFDecoder):
    """RFDecoder whose outputs are mapped into physically admissible ranges.

    Path losses are non-negative and bounded, RSSI lies in [min_rssi, max_rssi]
    (default [-150, 0] dBm) and coverage in [0, 1].
    """
    
    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 min_path_loss: float = 0.0,
                 max_path_loss: float = 200.0,
                 min_rssi: float = -150.0,
                 max_rssi: float = 0.0,
                 **kwargs):
        super().__init__(in_channels, hidden_channels, **kwargs)
        
        self.min_path_loss = min_path_loss
        self.max_path_loss = max_path_loss
        self.min_rssi = min_rssi
        self.max_rssi = max_rssi
    
    def forward(self, 
                terrain_embeddings: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Return bounded predictions [N, 5] (extra keyword arguments are ignored)."""
        raw_predictions = super().forward(terrain_embeddings, use_separate_heads=False)
        
        constrained = torch.zeros_like(raw_predictions)
        
        constrained[:, 0] = torch.clamp(
            F.softplus(raw_predictions[:, 0]),
            self.min_path_loss,
            self.max_path_loss
        )
        constrained[:, 1] = torch.clamp(
            F.softplus(raw_predictions[:, 1]),
            0,
            50  # vegetation loss capped at 50 dB
        )
        constrained[:, 2] = torch.clamp(
            F.softplus(raw_predictions[:, 2]),
            0,
            30  # terrain loss capped at 30 dB
        )

        rssi_range = self.max_rssi - self.min_rssi
        constrained[:, 3] = self.min_rssi + rssi_range * torch.sigmoid(raw_predictions[:, 3])

        # Coverage already went through the sigmoid in RFDecoder.forward.
        constrained[:, 4] = raw_predictions[:, 4]
        
        return constrained


def validate_decoder(decoder: RFDecoder,
                    embeddings: torch.Tensor,
                    verbose: bool = True) -> bool:
    """Run one forward pass; check the width, NaNs and, for the constrained decoder, the ranges."""
    errors = []
    
    try:
        decoder.eval()
        with torch.no_grad():
            predictions = decoder(embeddings)
        
        if predictions.shape[1] != 5:
            errors.append(f"Expected 5 outputs, got {predictions.shape[1]}")
        
        if torch.isnan(predictions).any():
            errors.append("Predictions contain NaN")
        
        if isinstance(decoder, PhysicsConstrainedDecoder):
            path_loss = predictions[:, 0]
            if (path_loss < 0).any():
                errors.append(f"path_loss < 0: min={path_loss.min():.2f}")
            
            rssi = predictions[:, 3]
            if rssi.min() < -200 or rssi.max() > 50:
                errors.append(f"RSSI out of range: [{rssi.min():.1f}, {rssi.max():.1f}]")
            
            coverage = predictions[:, 4]
            if coverage.min() < 0 or coverage.max() > 1:
                errors.append(f"coverage not in [0,1]: [{coverage.min():.3f}, {coverage.max():.3f}]")
        
    except Exception as e:
        errors.append(f"Forward pass failed: {str(e)}")
    
    if errors:
        for e in errors:
            print(f"❌ VALIDATE Error: {e}")
        return False
    
    if verbose:
        print(f"✅ validate_decoder PASSED")
        print(f"   Output shape: {predictions.shape}")
        print(f"   path_loss range: [{predictions[:, 0].min():.1f}, {predictions[:, 0].max():.1f}] dB")
        print(f"   rssi range: [{predictions[:, 3].min():.1f}, {predictions[:, 3].max():.1f}] dBm")
        print(f"   coverage range: [{predictions[:, 4].min():.3f}, {predictions[:, 4].max():.3f}]")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: RFDecoder")
    print("=" * 60)
    
    n_nodes = 500
    hidden_dim = 512
    embeddings = torch.randn(n_nodes, hidden_dim)
    
    print("\n📊 Testando RFDecoder básico...")
    decoder = RFDecoder(in_channels=hidden_dim)
    valid1 = validate_decoder(decoder, embeddings)
    
    print("\n📊 Testando PhysicsConstrainedDecoder...")
    decoder_physics = PhysicsConstrainedDecoder(in_channels=hidden_dim)
    valid2 = validate_decoder(decoder_physics, embeddings)
    
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n   Decoder parameters: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if (valid1 and valid2) else "❌ TESTE FALHOU")
    print("=" * 60)
