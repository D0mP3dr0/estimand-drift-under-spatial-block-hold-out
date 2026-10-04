"""Configuration package of the partition and training code: re-exports the dataclasses of config.py
and the global `config` instance."""
from .config import Config, config
from .config import TerrainConfig, AntennaConfig, RFOutputConfig
from .config import ModelConfig, TrainingConfig, PathConfig

__all__ = [
    'Config', 'config',
    'TerrainConfig', 'AntennaConfig', 'RFOutputConfig',
    'ModelConfig', 'TrainingConfig', 'PathConfig'
]
