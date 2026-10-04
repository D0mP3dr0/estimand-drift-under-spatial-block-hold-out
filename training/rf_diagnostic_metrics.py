"""Per-epoch diagnostic metrics of the radio-propagation predictions.

``RFDiagnosticMetrics`` precomputes, once per node set, the environmental
masks (vegetation, open area, water from NDVI/NDWI), the distance of every
terrain node to the nearest antenna, the distance rings and a free-space
path loss (FSPL) reference. ``compute(preds, targets)`` then returns one
dict of 52 floats (the keys of ``PHASE_MAP``, also listed in ``ALL_COLUMNS``)
grouped in five families: 1 distribution of the predictions, 2 physical
consistency, 3 error by spatial segment and coverage at several RSSI
thresholds, 4 feature-error correlations, 5 per-target error metrics.
The frozen GNN and MLP trainers use this class to add these columns to
their evaluation records.

Assumed terrain-feature columns (17): 0 elevation, 1 slope, 2 aspect,
3 curvature, 4 tpi, 5 tri, 6 roughness, 7 flow_acc, 8 B02, 9 B03, 10 B04,
11 B08 (NIR), 12 NDVI, 13 NDWI, 14 BSI, 15 shadow_idx, 16 has_lidar.
Target columns as named here: 0 path loss, 1 shadow margin, 2 diffraction,
3 RSSI, 4 coverage.

Usage (library module, no command line):
    diag = RFDiagnosticMetrics(terrain_pos, antenna_pos, terrain_features)
    row = diag.compute(preds, targets)   # once per epoch
It reads no files and writes none. When N differs from the number of
precomputed nodes, the subsample is taken on a fixed evenly spaced index
grid, so the result is deterministic.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Optional, Tuple


# RSSI thresholds (dBm) of the coverage metrics and distance-ring limits (m).
RSSI_THRESHOLDS_DBM   = [-85.0, -90.0, -95.0, -100.0]
DIST_NEAR_M           = 5_000.0
DIST_MID_M            = 15_000.0
FREQUENCY_MHZ         = 1_800.0  # default frequency of the FSPL reference
CHUNK_SIZE            = 100_000  # terrain nodes per chunk in the nearest-antenna search


def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation of two 1-D tensors; 0.0 for fewer than 5 values or zero variance."""
    if x.numel() < 5:
        return 0.0
    x = x.float().flatten()
    y = y.float().flatten()
    std_x = x.std()
    std_y = y.std()
    if std_x < 1e-8 or std_y < 1e-8:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    corr = (xc * yc).mean() / (std_x * std_y)
    return float(corr.clamp(-1.0, 1.0))


def _safe_p90(errors: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """90th percentile of the (optionally masked) errors; NaN when the selection is empty."""
    if mask is not None:
        errors = errors[mask]
    if errors.numel() == 0:
        return float("nan")
    return float(np.percentile(errors.numpy(), 90))


def _safe_mae(errors: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Mean of the (optionally masked) errors; NaN when the selection is empty."""
    if mask is not None:
        errors = errors[mask]
    if errors.numel() == 0:
        return float("nan")
    return float(errors.mean())


def _fspl_db(dist_m: torch.Tensor, freq_mhz: float = FREQUENCY_MHZ) -> torch.Tensor:
    """Free-space path loss in dB for distances in metres (clipped to at least 1 m).

    FSPL = 20 log10(d_km) + 20 log10(f_MHz) + 32.44.
    """
    d_km = dist_m.clamp(min=1.0) / 1000.0
    return 20.0 * torch.log10(d_km) + 20.0 * np.log10(freq_mhz) + 32.44


class RFDiagnosticMetrics:
    """Diagnostic metrics over a fixed set of terrain nodes.

    Masks, distances and the FSPL reference are computed once in ``__init__``;
    ``compute`` is cheap enough to be called every epoch.
    """

    # Metric name -> family (1 to 5); the keys are exactly the keys returned by compute().
    PHASE_MAP = {
        "rssi_pred_std": 1, "rssi_pred_p10": 1, "rssi_pred_p25": 1,
        "rssi_pred_p75": 1, "rssi_pred_p90": 1, "path_loss_pred_std": 1,
        "rssi_target_std": 1, "distribution_fidelity": 1,
        "physics_fspl_compliance": 2, "physics_rssi_range_ok": 2,
        "physics_distance_gradient": 2, "physics_shadowing_ndvi_corr": 2,
        "physics_diffraction_roughness_corr": 2, "physics_vegetation_compliance": 2,
        "physics_path_loss_bias": 2, "physics_rssi_bias": 2,
        "p90_ring_near": 3, "p90_ring_mid": 3, "p90_ring_far": 3,
        "mae_ring_near": 3, "mae_ring_mid": 3, "mae_ring_far": 3,
        "p90_vegetation": 3, "p90_open_area": 3, "p90_water": 3,
        "coverage_at_85dbm": 3, "coverage_at_90dbm": 3,
        "coverage_at_95dbm": 3, "coverage_at_100dbm": 3,
        "coverage_spread": 3,
        "corr_dist_rssi_error": 4, "corr_elevation_pl_error": 4,
        "corr_slope_pl_error": 4, "corr_ndvi_rssi_error": 4,
        "corr_roughness_pl_error": 4, "corr_has_lidar_error": 4,
        "corr_b08_error": 4, "corr_shadow_idx_rssi_error": 4,
        "pl_mae": 5, "pl_rmse": 5, "pl_bias": 5, "pl_p90": 5, "pl_std_pred": 5,
        "shadow_mae": 5, "shadow_std_pred": 5,
        "diffr_mae": 5, "diffr_std_pred": 5,
        "rssi_mae": 5, "rssi_rmse": 5, "rssi_bias": 5, "rssi_p90": 5,
        "rssi_std_pred": 5,
    }

    ALL_COLUMNS = list(PHASE_MAP.keys())

    def __init__(
        self,
        terrain_pos: torch.Tensor,
        antenna_pos: Optional[torch.Tensor],
        terrain_features: torch.Tensor,
        frequency_mhz: float = FREQUENCY_MHZ,
        dist_nearest_m: Optional[torch.Tensor] = None,
    ) -> None:
        """terrain_pos: [N, 2] (lon, lat) in degrees; antenna_pos: [M, 2] (lon, lat) or None;
        terrain_features: [N, >=17] in the column order of the module docstring;
        dist_nearest_m: optional precomputed [N] distance to the nearest antenna in metres.
        """
        self.freq_mhz = frequency_mhz
        n_feats = terrain_features.shape[1]

        f = terrain_features.float().cpu()
        self.elevation  = f[:, 0]
        self.slope      = f[:, 1]  if n_feats > 1  else None
        self.roughness  = f[:, 6]  if n_feats > 6  else None
        self.b08        = f[:, 11] if n_feats > 11 else None
        self.ndvi       = f[:, 12] if n_feats > 12 else None
        self.ndwi       = f[:, 13] if n_feats > 13 else None
        self.shadow_idx = f[:, 15] if n_feats > 15 else None
        self.has_lidar  = f[:, 16] if n_feats > 16 else None

        # Land-cover masks: vegetation NDVI > 0.5, water NDWI > 0.3, open area NDVI < 0.2.
        self.mask_veg   = (self.ndvi > 0.5)  if self.ndvi is not None else None
        self.mask_water = (self.ndwi > 0.3)  if self.ndwi is not None else None
        self.mask_open  = (self.ndvi < 0.2)  if self.ndvi is not None else None

        # Distance to the nearest antenna, in order of preference: the precomputed
        # dist_nearest_m, the haversine distance from terrain_pos/antenna_pos, else NaN.
        if dist_nearest_m is not None:
            self.dist_nearest = dist_nearest_m.float().cpu()
        elif antenna_pos is not None and antenna_pos.shape[0] > 0:
            self.dist_nearest = self._compute_nearest_dist(
                terrain_pos.float().cpu(),
                antenna_pos.float().cpu(),
            )
        else:
            self.dist_nearest = torch.full((terrain_pos.shape[0],), float("nan"))

        valid_dist = ~torch.isnan(self.dist_nearest)
        self.mask_near = valid_dist & (self.dist_nearest < DIST_NEAR_M)
        self.mask_mid  = valid_dist & (self.dist_nearest >= DIST_NEAR_M) & (self.dist_nearest < DIST_MID_M)
        self.mask_far  = valid_dist & (self.dist_nearest >= DIST_MID_M)

        dist_valid = self.dist_nearest.clone()
        # Placeholder distance where it is unknown; those nodes are excluded through has_dist.
        dist_valid[~valid_dist] = 1.0
        self.fspl_ref = _fspl_db(dist_valid, frequency_mhz)
        self.has_dist = valid_dist


    def compute(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Return the 52 metrics for predictions and targets of shape [N, 5].

        N may exceed the number of precomputed nodes (a neighbour sampler can repeat
        nodes). Families 1 and 5 use all N rows; families 2, 3 and 4, which pair rows
        with the precomputed masks and features, use an evenly spaced subsample of
        N rows of that size.
        """
        preds   = preds.float().cpu()
        targets = targets.float().cpu()

        n_mask  = self.mask_near.shape[0]
        n_preds = preds.shape[0]
        if n_preds != n_mask:
            # Evenly spaced indices; the clamp guards against float32 rounding at the end.
            idx = torch.linspace(0, n_preds - 1, n_mask).long().clamp(0, n_preds - 1)
            preds_spatial   = preds[idx]
            targets_spatial = targets[idx]
        else:
            preds_spatial   = preds
            targets_spatial = targets

        pl_pred     = preds[:, 0];   pl_tgt    = targets[:, 0]
        shadow_pred = preds[:, 1];   shadow_tgt = targets[:, 1]
        diffr_pred  = preds[:, 2];   diffr_tgt  = targets[:, 2]
        rssi_pred   = preds[:, 3];   rssi_tgt   = targets[:, 3]

        pl_pred_sp   = preds_spatial[:, 0];  pl_tgt_sp   = targets_spatial[:, 0]
        rssi_pred_sp = preds_spatial[:, 3];  rssi_tgt_sp = targets_spatial[:, 3]
        shadow_pred_sp = preds_spatial[:, 1]
        diffr_pred_sp  = preds_spatial[:, 2]

        metrics: Dict[str, float] = {}
        metrics.update(self._phase1_distribution(rssi_pred, rssi_tgt, pl_pred))
        metrics.update(self._phase4_correlations(pl_pred_sp, pl_tgt_sp,
                                                  rssi_pred_sp, rssi_tgt_sp))
        metrics.update(self._phase5_per_target(pl_pred, pl_tgt, shadow_pred, shadow_tgt,
                                                diffr_pred, diffr_tgt, rssi_pred, rssi_tgt))
        metrics.update(self._phase2_physics(pl_pred_sp, pl_tgt_sp,
                                             rssi_pred_sp, rssi_tgt_sp,
                                             shadow_pred_sp, diffr_pred_sp))
        metrics.update(self._phase3_spatial(pl_pred_sp, pl_tgt_sp, rssi_pred_sp))
        return metrics


    def _phase1_distribution(
        self,
        rssi_pred: torch.Tensor,
        rssi_tgt: torch.Tensor,
        pl_pred: torch.Tensor,
    ) -> Dict[str, float]:
        """Spread and percentiles of the predicted RSSI; distribution_fidelity = std(pred) / std(target)."""
        rp = rssi_pred.numpy()
        tgt_std = float(rssi_tgt.std()) if rssi_tgt.std() > 1e-8 else 1.0
        pred_std = float(rssi_pred.std())
        return {
            "rssi_pred_std":      pred_std,
            "rssi_pred_p10":      float(np.percentile(rp, 10)),
            "rssi_pred_p25":      float(np.percentile(rp, 25)),
            "rssi_pred_p75":      float(np.percentile(rp, 75)),
            "rssi_pred_p90":      float(np.percentile(rp, 90)),
            "path_loss_pred_std": float(pl_pred.std()),
            "rssi_target_std":    tgt_std,
            "distribution_fidelity": pred_std / tgt_std,
        }


    def _phase2_physics(
        self,
        pl_pred: torch.Tensor, pl_tgt: torch.Tensor,
        rssi_pred: torch.Tensor, rssi_tgt: torch.Tensor,
        shadow_pred: torch.Tensor,
        diffr_pred: torch.Tensor,
    ) -> Dict[str, float]:
        """Physical consistency of the predictions.

        FSPL compliance (share with predicted path loss >= FSPL), share of RSSI in
        [-140, -30] dBm, correlation of RSSI with minus distance, of column 1 with NDVI
        and of column 2 with roughness, vegetated-to-overall RSSI MAE ratio, mean biases.
        """
        m: Dict[str, float] = {}

        if self.has_dist.any():
            compliant = (pl_pred[self.has_dist] >= self.fspl_ref[self.has_dist])
            m["physics_fspl_compliance"] = float(compliant.float().mean())
        else:
            m["physics_fspl_compliance"] = float("nan")

        range_ok = ((rssi_pred >= -140.0) & (rssi_pred <= -30.0))
        m["physics_rssi_range_ok"] = float(range_ok.float().mean())

        if self.has_dist.any():
            m["physics_distance_gradient"] = _pearsonr(
                rssi_pred[self.has_dist],
                -self.dist_nearest[self.has_dist],
            )
        else:
            m["physics_distance_gradient"] = float("nan")

        if self.ndvi is not None:
            m["physics_shadowing_ndvi_corr"] = _pearsonr(shadow_pred, self.ndvi)
        else:
            m["physics_shadowing_ndvi_corr"] = float("nan")

        if self.roughness is not None:
            m["physics_diffraction_roughness_corr"] = _pearsonr(diffr_pred, self.roughness)
        else:
            m["physics_diffraction_roughness_corr"] = float("nan")

        if self.mask_veg is not None and self.mask_veg.any():
            mae_veg = float(torch.abs(rssi_pred[self.mask_veg] - rssi_tgt[self.mask_veg]).mean())
            mae_all = float(torch.abs(rssi_pred - rssi_tgt).mean())
            m["physics_vegetation_compliance"] = mae_veg / (mae_all + 1e-8)
        else:
            m["physics_vegetation_compliance"] = float("nan")

        m["physics_path_loss_bias"] = float((pl_pred - pl_tgt).mean())
        m["physics_rssi_bias"]      = float((rssi_pred - rssi_tgt).mean())

        return m


    def _phase3_spatial(
        self,
        pl_pred: torch.Tensor, pl_tgt: torch.Tensor,
        rssi_pred: torch.Tensor,
    ) -> Dict[str, float]:
        """Path-loss error by distance ring and land cover, and coverage (% of nodes)
        above each RSSI threshold.
        """
        m: Dict[str, float] = {}
        pl_err = torch.abs(pl_pred - pl_tgt)

        for tag, mask in [("near", self.mask_near),
                          ("mid",  self.mask_mid),
                          ("far",  self.mask_far)]:
            m[f"p90_ring_{tag}"] = _safe_p90(pl_err, mask)
            m[f"mae_ring_{tag}"] = _safe_mae(pl_err, mask)

        for tag, mask in [("vegetation", self.mask_veg),
                          ("open_area",  self.mask_open),
                          ("water",      self.mask_water)]:
            m[f"p90_{tag}"] = _safe_p90(pl_err, mask)

        for thr in RSSI_THRESHOLDS_DBM:
            key = f"coverage_at_{abs(int(thr))}dbm"
            m[key] = float((rssi_pred > thr).float().mean() * 100.0)

        # Spread between the -85 dBm and -95 dBm coverage levels, in percentage points.
        m["coverage_spread"] = (
            m["coverage_at_85dbm"] - m["coverage_at_95dbm"]
        )

        return m


    def _phase4_correlations(
        self,
        pl_pred: torch.Tensor, pl_tgt: torch.Tensor,
        rssi_pred: torch.Tensor, rssi_tgt: torch.Tensor,
    ) -> Dict[str, float]:
        """Pearson correlation between the absolute errors and distance or terrain features."""
        m: Dict[str, float] = {}
        pl_err   = torch.abs(pl_pred - pl_tgt)
        rssi_err = torch.abs(rssi_pred - rssi_tgt)

        if self.has_dist.any():
            m["corr_dist_rssi_error"] = _pearsonr(
                self.dist_nearest[self.has_dist], rssi_err[self.has_dist]
            )
        else:
            m["corr_dist_rssi_error"] = float("nan")

        for feat_name, feat_tensor, err_tensor in [
            ("elevation_pl",   self.elevation, pl_err),
            ("slope_pl",       self.slope,     pl_err),
            ("ndvi_rssi",      self.ndvi,      rssi_err),
            ("roughness_pl",   self.roughness, pl_err),
            ("has_lidar",      self.has_lidar, rssi_err),
            ("b08",            self.b08,       rssi_err),
            ("shadow_idx_rssi",self.shadow_idx, rssi_err),
        ]:
            if feat_tensor is not None:
                m[f"corr_{feat_name}_error"] = _pearsonr(feat_tensor, err_tensor)
            else:
                m[f"corr_{feat_name}_error"] = float("nan")

        return m


    def _phase5_per_target(
        self,
        pl_pred: torch.Tensor,     pl_tgt: torch.Tensor,
        shadow_pred: torch.Tensor, shadow_tgt: torch.Tensor,
        diffr_pred: torch.Tensor,  diffr_tgt: torch.Tensor,
        rssi_pred: torch.Tensor,   rssi_tgt: torch.Tensor,
    ) -> Dict[str, float]:
        """MAE, RMSE, bias, P90 and prediction std for path loss and RSSI; MAE and
        prediction std for columns 1 and 2.
        """
        m: Dict[str, float] = {}

        def _metrics(pred: torch.Tensor, tgt: torch.Tensor, prefix: str) -> None:
            err = torch.abs(pred - tgt)
            bias = float((pred - tgt).mean())
            m[f"{prefix}_mae"]      = float(err.mean())
            m[f"{prefix}_rmse"]     = float(torch.sqrt((err ** 2).mean()))
            m[f"{prefix}_bias"]     = bias
            m[f"{prefix}_p90"]      = float(np.percentile(err.numpy(), 90))
            m[f"{prefix}_std_pred"] = float(pred.std())

        _metrics(pl_pred, pl_tgt, "pl")
        _metrics(rssi_pred, rssi_tgt, "rssi")

        for pred_t, tgt_t, prefix in [(shadow_pred, shadow_tgt, "shadow"),
                                       (diffr_pred,  diffr_tgt,  "diffr")]:
            err = torch.abs(pred_t - tgt_t)
            m[f"{prefix}_mae"]      = float(err.mean())
            m[f"{prefix}_std_pred"] = float(pred_t.std())

        return m


    @staticmethod
    def _compute_nearest_dist(
        terrain_pos: torch.Tensor,
        antenna_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Haversine distance (m) from each terrain node to the nearest antenna.

        Positions are (lon, lat) in degrees; terrain nodes are processed in chunks of
        CHUNK_SIZE to bound memory.
        """
        import numpy as np

        N   = terrain_pos.shape[0]
        R   = 6_371_000.0  # mean Earth radius (m)

        ter_lon = terrain_pos[:, 0].numpy()
        ter_lat = terrain_pos[:, 1].numpy()
        ant_lon = antenna_pos[:, 0].numpy()
        ant_lat = antenna_pos[:, 1].numpy()

        φ2 = np.radians(ant_lat)
        λ2 = np.radians(ant_lon)

        dist_np = np.empty(N, dtype=np.float32)
        for start in range(0, N, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, N)
            φ1 = np.radians(ter_lat[start:end])[:, None]
            λ1 = np.radians(ter_lon[start:end])[:, None]

            dφ = φ2[None, :] - φ1
            dλ = λ2[None, :] - λ1
            a  = (np.sin(dφ / 2) ** 2
                  + np.cos(φ1) * np.cos(φ2[None, :]) * np.sin(dλ / 2) ** 2)
            d  = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            dist_np[start:end] = d.min(axis=1)

        return torch.from_numpy(dist_np)
