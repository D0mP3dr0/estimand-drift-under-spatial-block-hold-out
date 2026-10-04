"""Central configuration of the GNN-RF code: feature and target layouts, model and training defaults.

Dataclasses describe the terrain node features (17 inputs, 14 terrain targets, 2 edge
features), the antenna features (10), the RF outputs (5 channels: total, vegetation and terrain
path loss in dB, RSSI in dBm, coverage probability), the model and training defaults, and the
project paths of the original machine. A global instance `config` is created at import time;
`spatial_cv.py` and the training modules import it. These are defaults: the launchers in
results/gpu/ pass their own settings (hidden size, learning rate, batch size) on the command line.

Usage: import `config`; `python config.py` runs `Config.validate()` and prints a summary.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import torch


@dataclass
class TerrainConfig:
    """Terrain node features, terrain targets and edge features (index order matters)."""

    input_dim: int = 17
    feature_names: List[str] = field(default_factory=lambda: [
        'elevation',      # 0: SRTM/LiDAR, z-score
        'slope',          # 1: [0, 1], i.e. 0-90 degrees
        'aspect_cos',     # 2: [-1, 1]
        'aspect_sin',     # 3: [-1, 1]
        'curvature',      # 4: raw
        'tpi',            # 5: z-score
        'tri',            # 6: z-score
        'roughness',      # 7: z-score
        'b02_norm',       # 8: Sentinel-2 blue, [0, 1]
        'b03_norm',       # 9: Sentinel-2 green, [0, 1]
        'b04_norm',       # 10: Sentinel-2 red, [0, 1]
        'b08_norm',       # 11: Sentinel-2 NIR, [0, 1]
        'ndvi',           # 12: [-1, 1]
        'ndwi',           # 13: [-1, 1]
        'has_lidar',      # 14: 0/1
        'z_lidar',        # 15: metres
        'confidence'      # 16: [0, 1]
    ])

    output_dim: int = 14
    target_names: List[str] = field(default_factory=lambda: [
        'elevation',      # 0
        'slope',          # 1
        'aspect_cos',     # 2
        'aspect_sin',     # 3
        'curvature',      # 4
        'tpi',            # 5
        'tri',            # 6
        'roughness',      # 7
        'flow_accum',     # 8
        'canopy',         # 9: canopy height (see canopy_index)
        'ndvi',           # 10
        'ndwi',           # 11
        'bsi',            # 12
        'shadow'          # 13
    ])

    canopy_index: int = 9

    edge_dim: int = 2
    edge_feature_names: List[str] = field(default_factory=lambda: [
        'dist_norm',      # normalised distance, [0, 1]
        'dz_norm'         # elevation difference, z-score
    ])


@dataclass
class AntennaConfig:
    """Antenna (transmitter) features and their normalisation ranges."""

    input_dim: int = 10
    feature_names: List[str] = field(default_factory=lambda: [
        'lat',              # 0: [-90, 90]
        'lon',              # 1: [-180, 180]
        'altura_antena',    # 2: antenna height, [0, 200] m
        'freq_tx_mhz',      # 3: [700, 6000] MHz
        'potencia_watts',   # 4: transmit power, log-scaled
        'ganho_antena',     # 5: antenna gain, [0, 30] dBi
        'azimute_sin',      # 6: [-1, 1]
        'azimute_cos',      # 7: [-1, 1]
        'tilt',             # 8: [-20, 90] degrees
        'tecnologia_enc'    # 9: encoded technology
    ])

    freq_range: tuple = (700, 6000)       # MHz
    potencia_range: tuple = (0.1, 10000)  # W
    altura_range: tuple = (0, 200)        # m
    ganho_range: tuple = (0, 30)          # dBi


@dataclass
class RFOutputConfig:
    """RF output channels, physical ranges and RSSI coverage classes."""

    output_dim: int = 5
    target_names: List[str] = field(default_factory=lambda: [
        'path_loss_total',      # 0: dB
        'path_loss_vegetation', # 1: dB
        'path_loss_terrain',    # 2: dB
        'rssi',                 # 3: dBm
        'coverage_prob'         # 4: [0, 1]
    ])

    path_loss_range: tuple = (0, 200)     # dB
    rssi_range: tuple = (-150, 0)         # dBm

    # Coverage classes by RSSI interval (dBm): 1 excellent ... 5 no coverage (below -110 dBm).
    coverage_classes: Dict[int, tuple] = field(default_factory=lambda: {
        1: (-70, float('inf')),
        2: (-85, -70),
        3: (-100, -85),
        4: (-110, -100),
        5: (float('-inf'), -110)
    })


@dataclass
class ModelConfig:
    """Default GNN dimensions, attention heads, dropout and activation."""

    hidden_dim: int = 512
    num_layers: int = 4
    heads: int = 4
    dropout: float = 0.1

    edge_dim: int = 2

    activation: str = 'leaky_relu'
    negative_slope: float = 0.2


@dataclass
class TrainingConfig:
    """Default optimisation, curriculum, early-stopping and spatial-CV settings."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    chunk_size: int = 100_000

    # Curriculum of 5 phases with these epoch counts.
    num_phases: int = 5
    epochs_per_phase: List[int] = field(default_factory=lambda: [20, 25, 25, 20, 15])

    patience: int = 10
    min_delta: float = 1e-4

    num_folds: int = 5
    buffer_km: float = 2.0


@dataclass
class PathConfig:
    """Project paths on the original machine (not used by the scripts of this repository)."""

    base_dir: Path = Path("f:/arpia_topo_refinado/TOPO_RF/GNN_RF_V2")

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "01_data"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "02_models"

    @property
    def training_dir(self) -> Path:
        return self.base_dir / "03_training"

    @property
    def baselines_dir(self) -> Path:
        return self.base_dir / "04_baselines"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "05_outputs"

    @property
    def evaluation_dir(self) -> Path:
        return self.base_dir / "06_evaluation"

    @property
    def paper_dir(self) -> Path:
        return self.base_dir / "07_paper"

    @property
    def checkpoints_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    gnn_topo_checkpoint: Path = Path("f:/arpia_topo_refinado/v21/checkpoints")
    gnn_topo_dataset: Path = Path("f:/GRAPH_V18.3")
    anatel_database: Path = Path("f:/arpia_topo_refinado/TOPO_RF/GRAFO")


@dataclass
class Config:
    """Top-level configuration grouping all sections; device is CUDA when available."""

    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    rf_output: RFOutputConfig = field(default_factory=RFOutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> bool:
        """Check the fixed dimensions (17 inputs, 14 targets, 2 edge features, canopy index 9, hidden 512)."""
        errors = []

        if self.terrain.input_dim != 17:
            errors.append(f"terrain.input_dim deve ser 17, got {self.terrain.input_dim}")

        if self.terrain.output_dim != 14:
            errors.append(f"terrain.output_dim deve ser 14, got {self.terrain.output_dim}")

        if self.terrain.edge_dim != 2:
            errors.append(f"terrain.edge_dim deve ser 2, got {self.terrain.edge_dim}")

        if self.terrain.canopy_index != 9:
            errors.append(f"terrain.canopy_index deve ser 9, got {self.terrain.canopy_index}")

        if self.model.hidden_dim != 512:
            errors.append(f"model.hidden_dim deve ser 512, got {self.model.hidden_dim}")

        if errors:
            for e in errors:
                print(f"❌ Config Error: {e}")
            return False

        print("✅ Config validation PASSED")
        return True


config = Config()


if __name__ == "__main__":
    config.validate()

    print(f"\n📊 Configuração GNN-RF Propagation V2")
    print(f"   Terrain features: {config.terrain.input_dim}")
    print(f"   Terrain targets: {config.terrain.output_dim}")
    print(f"   Canopy index: {config.terrain.canopy_index}")
    print(f"   Antenna features: {config.antenna.input_dim}")
    print(f"   RF outputs: {config.rf_output.output_dim}")
    print(f"   Hidden dim: {config.model.hidden_dim}")
    print(f"   Device: {config.device}")
