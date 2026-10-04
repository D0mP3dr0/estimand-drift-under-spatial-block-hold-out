"""Physics-informed training losses of the radio-propagation models.

Predictions and targets are [N, 5] tensors with columns
0 path_loss_total (dB), 1 path_loss_vegetation (dB), 2 path_loss_terrain (dB),
3 rssi (dBm), 4 coverage.

``PhysicsRFLoss``: per-column Huber (or MSE) loss combined with softmax
weights over five learnable log-weights, plus two optional constraint terms,
a simplified vegetation-attenuation prior (in the spirit of ITU-R P.833) and
a free-space path loss (FSPL) lower bound on the total path loss.

``CurriculumRFLoss`` (the class used by the frozen GNN and MLP trainers)
adds a per-column curriculum mask and three penalties on the predicted
RSSI and on column 1: a distance-gradient penalty (RSSI should decrease with
distance to the nearest antenna), a variance penalty (the spread of the
predicted RSSI should not fall below that of the target), and a correlation
penalty between column 1 and NDVI. The trainers set the weights of these
penalties from their command-line arguments.

The module reads no files and writes none. It expects the original module
layout, with the ``config`` package in the parent directory of this file.
Determinism: the distance-gradient penalty draws random node pairs with
``torch.randperm``, so it follows the global torch seed set by the trainer.

Usage: run the file directly for a smoke test on random tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


class PhysicsRFLoss(nn.Module):
    """Weighted per-target regression loss plus optional vegetation and FSPL constraints."""
    
    def __init__(self,
                 frequency_mhz: float = 900.0,
                 use_huber: bool = True,
                 huber_delta: float = 5.0,
                 vegetation_weight: float = 0.1,
                 fspl_weight: float = 0.05,
                 learnable_weights: bool = True):
        super().__init__()
        
        self.frequency_mhz = frequency_mhz
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.vegetation_weight = vegetation_weight
        self.fspl_weight = fspl_weight
        
        if learnable_weights:
            # Weights are softmax(log_weights), so they stay positive and sum to one.
            self.log_weights = nn.Parameter(torch.zeros(5))
        else:
            self.register_buffer('log_weights', torch.zeros(5))
        
        self.target_names = [
            'path_loss_total',
            'path_loss_vegetation',
            'path_loss_terrain',
            'rssi',
            'coverage'
        ]
    
    @property
    def weights(self) -> torch.Tensor:
        return F.softmax(self.log_weights, dim=0)
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                canopy_height: Optional[torch.Tensor] = None,
                distances: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return (total_loss, loss_dict).

        predictions, targets: [N, 5]; canopy_height: [N] in metres (enables the
        vegetation term); distances: [N] transmitter-receiver distance in metres
        (enables the FSPL term).
        """
        weights = self.weights
        loss_dict = {}
        
        for i, name in enumerate(self.target_names):
            pred = predictions[:, i]
            true = targets[:, i]
            
            if self.use_huber:
                loss = F.huber_loss(pred, true, delta=self.huber_delta, reduction='mean')
            else:
                loss = F.mse_loss(pred, true, reduction='mean')
            
            loss_dict[f'loss_{name}'] = loss
        
        prediction_loss = sum(
            weights[i] * loss_dict[f'loss_{self.target_names[i]}']
            for i in range(5)
        )
        loss_dict['prediction_loss'] = prediction_loss
        
        constraint_loss = torch.tensor(0.0, device=predictions.device)
        
        if canopy_height is not None:
            veg_constraint = self._vegetation_constraint(
                predictions[:, 1],
                canopy_height
            )
            loss_dict['constraint_vegetation'] = veg_constraint
            constraint_loss = constraint_loss + self.vegetation_weight * veg_constraint
        
        if distances is not None:
            fspl_constraint = self._fspl_constraint(
                predictions[:, 0],
                distances
            )
            loss_dict['constraint_fspl'] = fspl_constraint
            constraint_loss = constraint_loss + self.fspl_weight * fspl_constraint
        
        loss_dict['constraint_loss'] = constraint_loss
        
        total_loss = prediction_loss + constraint_loss
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def _vegetation_constraint(self,
                              pred_veg_loss: torch.Tensor,
                              canopy_height: torch.Tensor
                              ) -> torch.Tensor:
        """Huber distance (delta 3 dB) between the predicted vegetation loss and a*f^b*h.

        f is the frequency in GHz and h the canopy height clipped to [0, 30] m, used
        as the path length through vegetation; a = 0.2 and b = 0.3.
        """
        a = 0.2
        b = 0.3
        freq_ghz = self.frequency_mhz / 1000.0
        
        expected_loss = a * (freq_ghz ** b) * torch.clamp(canopy_height, 0, 30)
        
        constraint = F.huber_loss(pred_veg_loss, expected_loss, delta=3.0)
        
        return constraint
    
    def _fspl_constraint(self,
                        pred_total_loss: torch.Tensor,
                        distances: torch.Tensor
                        ) -> torch.Tensor:
        """Mean hinge penalty relu(FSPL - predicted total path loss).

        FSPL(dB) = 20 log10(d_km) + 20 log10(f_MHz) + 32.44.
        """
        import math
        
        d_km = distances / 1000.0 + 1e-6  # metres to km; the offset avoids log10(0)
        fspl = 20 * torch.log10(d_km) + 20 * math.log10(self.frequency_mhz) + 32.44
        
        violation = F.relu(fspl - pred_total_loss)
        
        return violation.mean()


class CurriculumRFLoss(PhysicsRFLoss):
    """PhysicsRFLoss with a curriculum mask over the five targets and three extra penalties.

    Only the masked targets enter the returned loss; ``set_phase`` sets the mask
    and the scale of the constraint terms.
    """

    def __init__(
        self,
        distance_gradient_weight: float = 0.05,
        distance_gradient_n_pairs: int = 512,
        variance_weight: float = 0.02,
        shadowing_ndvi_weight: float = 0.03,
        shadowing_ndvi_min_corr: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.distance_gradient_weight  = distance_gradient_weight
        self.distance_gradient_n_pairs = distance_gradient_n_pairs
        self.variance_weight           = variance_weight
        self.shadowing_ndvi_weight     = shadowing_ndvi_weight
        self.shadowing_ndvi_min_corr   = shadowing_ndvi_min_corr

        self.register_buffer('target_mask', torch.ones(5))

        self.constraint_scale = 1.0
    
    def _distance_gradient_penalty(
        self,
        pred_rssi: torch.Tensor,
        dist_to_ant: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise ordering penalty on predicted RSSI versus distance to the nearest antenna.

        Draws K = min(n_pairs, N // 2) disjoint random pairs (i, j); over the pairs with
        dist_i < dist_j returns mean(relu(rssi_j - rssi_i + 1 dB)). Zero when K < 4.
        """
        N = pred_rssi.shape[0]
        K = min(self.distance_gradient_n_pairs, N // 2)
        if K < 4:
            return torch.tensor(0.0, device=pred_rssi.device)

        idx = torch.randperm(N, device=pred_rssi.device)[:K * 2]
        i_idx = idx[:K]
        j_idx = idx[K:]

        dist_i = dist_to_ant[i_idx]
        dist_j = dist_to_ant[j_idx]
        rssi_i = pred_rssi[i_idx]
        rssi_j = pred_rssi[j_idx]

        closer = dist_i < dist_j
        if closer.sum() == 0:
            return torch.tensor(0.0, device=pred_rssi.device)

        margin = 1.0
        violation = torch.relu(rssi_j[closer] - rssi_i[closer] + margin)
        return violation.mean()


    def _variance_penalty(
        self,
        pred_rssi: torch.Tensor,
        tgt_rssi: torch.Tensor,
    ) -> torch.Tensor:
        """Return relu(std(target RSSI) - std(predicted RSSI))**2.

        Zero when the predictions are at least as dispersed as the targets; the target
        standard deviation carries no gradient. Counteracts regression to the mean.
        """
        pred_std = pred_rssi.std()
        tgt_std  = tgt_rssi.std().detach()
        if tgt_std < 1e-4:
            return torch.tensor(0.0, device=pred_rssi.device)
        return torch.relu(tgt_std - pred_std) ** 2


    def _shadowing_ndvi_penalty(
        self,
        shadow_pred: torch.Tensor,
        ndvi_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return relu(min_corr - corr(column-1 prediction, NDVI)).

        Pearson correlation computed differentiably through the prediction only (NDVI
        is detached). Zero for fewer than 10 nodes or a constant NDVI.
        """
        n = shadow_pred.shape[0]
        if n < 10:
            return torch.tensor(0.0, device=shadow_pred.device)

        nf = ndvi_feat.detach().float()
        if nf.std() < 1e-6:
            return torch.tensor(0.0, device=shadow_pred.device)

        sp = shadow_pred - shadow_pred.mean()
        nf_c = nf - nf.mean()
        std_s = shadow_pred.std().clamp(min=1e-6)
        std_n = nf.std().clamp(min=1e-6)
        corr  = (sp * nf_c).mean() / (std_s * std_n)
        return torch.relu(self.shadowing_ndvi_min_corr - corr)

    def set_phase(self, phase: int) -> None:
        """Set the target mask and constraint scale of a curriculum phase.

        Phase 1: total path loss only; 2: + terrain loss (both with constraint scale 0);
        3: + vegetation loss (scale 0.5); 4: + RSSI (0.8); 5 or higher: all five (1.0).
        """
        if phase == 1:
            self.target_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
            self.constraint_scale = 0.0
        elif phase == 2:
            self.target_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
            self.constraint_scale = 0.0
        elif phase == 3:
            self.target_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
            self.constraint_scale = 0.5
        elif phase == 4:
            self.target_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
            self.constraint_scale = 0.8
        else:
            self.target_mask = torch.ones(5)
            self.constraint_scale = 1.0
    
    def forward(
        self,
        predictions,
        targets,
        dist_to_ant: Optional[torch.Tensor] = None,
        ndvi: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Curriculum-masked loss plus the physics penalties; returns (masked_loss, loss_dict).

        dist_to_ant: [N] distance in metres to the nearest antenna (enables the
        distance-gradient penalty on RSSI, column 3). ndvi: [N] NDVI per node (enables
        the NDVI correlation penalty while column 1 is active, phase 3 or higher).
        Other keyword arguments go to PhysicsRFLoss.forward.
        """
        total_loss, loss_dict = super().forward(predictions, targets, **kwargs)

        masked_loss = torch.tensor(0.0, device=predictions.device)
        for i, name in enumerate(self.target_names):
            if self.target_mask[i] > 0:
                masked_loss = masked_loss + self.weights[i] * loss_dict[f'loss_{name}']

        if 'constraint_loss' in loss_dict:
            masked_loss = masked_loss + self.constraint_scale * loss_dict['constraint_loss']

        pred_rssi = predictions[:, 3]

        dist_pen = torch.tensor(0.0, device=predictions.device)
        if dist_to_ant is not None and self.distance_gradient_weight > 0:
            dist_pen = self._distance_gradient_penalty(pred_rssi, dist_to_ant)
            masked_loss = masked_loss + self.distance_gradient_weight * dist_pen

        var_pen = torch.tensor(0.0, device=predictions.device)
        if self.variance_weight > 0:
            var_pen = self._variance_penalty(pred_rssi, targets[:, 3])
            masked_loss = masked_loss + self.variance_weight * var_pen

        ndvi_pen = torch.tensor(0.0, device=predictions.device)
        # Column 1 is the vegetation (shadowing) channel; the NDVI penalty applies only when it is unmasked.
        shadow_active = (self.target_mask[1] > 0)
        if ndvi is not None and shadow_active and self.shadowing_ndvi_weight > 0:
            shadow_pred = predictions[:, 1]
            ndvi_pen    = self._shadowing_ndvi_penalty(shadow_pred, ndvi)
            masked_loss = masked_loss + self.shadowing_ndvi_weight * ndvi_pen

        loss_dict['dist_gradient_penalty'] = dist_pen
        loss_dict['variance_penalty']      = var_pen
        loss_dict['shadowing_ndvi_penalty'] = ndvi_pen
        loss_dict['masked_loss']           = masked_loss

        return masked_loss, loss_dict


def validate_physics_loss(loss_fn: PhysicsRFLoss,
                         verbose: bool = True) -> bool:
    """Evaluate the loss on 100 random nodes; check it is a finite scalar with the expected keys."""
    errors = []
    
    try:
        n = 100
        predictions = torch.randn(n, 5)
        targets = torch.randn(n, 5)
        canopy_height = torch.rand(n) * 30
        distances = torch.rand(n) * 10000
        
        total_loss, loss_dict = loss_fn(
            predictions, targets,
            canopy_height=canopy_height,
            distances=distances
        )
        
        if total_loss.dim() != 0:
            errors.append(f"Loss should be scalar, got shape {total_loss.shape}")
        
        if torch.isnan(total_loss):
            errors.append("Loss is NaN")
        
        if torch.isinf(total_loss):
            errors.append("Loss is infinite")
        
        required_keys = ['total_loss', 'prediction_loss']
        for key in required_keys:
            if key not in loss_dict:
                errors.append(f"Missing key: {key}")
        
    except Exception as e:
        errors.append(f"Loss computation failed: {str(e)}")
    
    if errors:
        for e in errors:
            print(f"❌ VALIDATE Error: {e}")
        return False
    
    if verbose:
        print(f"✅ validate_physics_loss PASSED")
        print(f"   Total loss: {total_loss.item():.4f}")
        if 'constraint_vegetation' in loss_dict:
            print(f"   Vegetation constraint: {loss_dict['constraint_vegetation'].item():.4f}")
        if 'constraint_fspl' in loss_dict:
            print(f"   FSPL constraint: {loss_dict['constraint_fspl'].item():.4f}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: PhysicsRFLoss")
    print("=" * 60)
    
    print("\n📊 Testando PhysicsRFLoss...")
    loss_fn = PhysicsRFLoss(frequency_mhz=900.0)
    valid1 = validate_physics_loss(loss_fn)
    
    print("\n📊 Testando CurriculumRFLoss...")
    curriculum_loss = CurriculumRFLoss(frequency_mhz=1800.0)
    
    for phase in [1, 2, 3, 4, 5]:
        curriculum_loss.set_phase(phase)
        print(f"   Phase {phase}: mask={curriculum_loss.target_mask.tolist()}, scale={curriculum_loss.constraint_scale}")
    
    valid2 = validate_physics_loss(curriculum_loss)
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if (valid1 and valid2) else "❌ TESTE FALHOU")
    print("=" * 60)
