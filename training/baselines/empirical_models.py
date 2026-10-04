"""Empirical RF path-loss models used as analytical baselines.

Implements free-space path loss (Friis), Okumura-Hata (urban, suburban, rural/open
area), COST-231 Hata and a generic log-distance model, plus a small wrapper class that
turns path loss into received power. All functions take distances in metres and the
carrier frequency in MHz and return path loss in dB; distances are floored at 1 m
(0.001 km) before the logarithm. The frozen trainers (`training/frozen/`) import
`free_space_path_loss`, `okumura_hata_rural` and `cost231_hata` (suburban) to score the
analytical baselines next to the trained models.

The module inserts its parent folder in `sys.path` and imports the `config` package from
there (original layout: `config/` beside `04_baselines/`; here `partition/config/`).
Running it directly prints a path-loss table at 900 MHz for a few distances; it reads
and writes no file.

Usage: python empirical_models.py   (self-test only; normally imported)
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


def free_space_path_loss(distance_m: np.ndarray,
                        frequency_mhz: float) -> np.ndarray:
    """Free-space path loss (Friis): FSPL = 20 log10(d_km) + 20 log10(f_MHz) + 32.44, in dB."""
    d_km = np.maximum(distance_m / 1000.0, 0.001)
    fspl = 20 * np.log10(d_km) + 20 * np.log10(frequency_mhz) + 32.44
    return fspl


def okumura_hata_urban(distance_m: np.ndarray,
                       frequency_mhz: float,
                       h_tx: float = 30.0,
                       h_rx: float = 1.5) -> np.ndarray:
    """Okumura-Hata path loss for an urban area, in dB (nominal range 150-1500 MHz, 1-20 km).

    h_tx and h_rx are the transmitter and receiver antenna heights in metres; the receiver
    height correction is the large-city form, switched at 300 MHz.
    """
    f = frequency_mhz
    d = np.maximum(distance_m / 1000.0, 0.001)
    
    if f <= 300:
        a_hr = 8.29 * (np.log10(1.54 * h_rx))**2 - 1.1
    else:
        a_hr = 3.2 * (np.log10(11.75 * h_rx))**2 - 4.97
    
    L = (69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_tx) - a_hr +
         (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d))
    
    return L


def okumura_hata_suburban(distance_m: np.ndarray,
                          frequency_mhz: float,
                          h_tx: float = 30.0,
                          h_rx: float = 1.5) -> np.ndarray:
    """Okumura-Hata suburban path loss: urban value - 2 (log10(f/28))^2 - 5.4, in dB."""
    L_urban = okumura_hata_urban(distance_m, frequency_mhz, h_tx, h_rx)
    f = frequency_mhz
    
    L = L_urban - 2 * (np.log10(f / 28))**2 - 5.4
    
    return L


def okumura_hata_rural(distance_m: np.ndarray,
                       frequency_mhz: float,
                       h_tx: float = 30.0,
                       h_rx: float = 1.5) -> np.ndarray:
    """Okumura-Hata rural (open area) path loss: urban value - 4.78 (log10 f)^2 + 18.33 log10 f - 40.94, in dB."""
    L_urban = okumura_hata_urban(distance_m, frequency_mhz, h_tx, h_rx)
    f = frequency_mhz
    
    L = L_urban - 4.78 * (np.log10(f))**2 + 18.33 * np.log10(f) - 40.94
    
    return L


def cost231_hata(distance_m: np.ndarray,
                 frequency_mhz: float,
                 h_tx: float = 30.0,
                 h_rx: float = 1.5,
                 environment: str = 'urban') -> np.ndarray:
    """COST-231 Hata path loss in dB (extension of Hata to 1500-2000 MHz).

    Uses the small/medium-city receiver height correction; the environment constant C_m is
    3 dB for 'urban' and 0 dB otherwise.
    """
    f = frequency_mhz
    d = np.maximum(distance_m / 1000.0, 0.001)
    
    a_hr = (1.1 * np.log10(f) - 0.7) * h_rx - (1.56 * np.log10(f) - 0.8)
    
    C_m = 3.0 if environment == 'urban' else 0.0
    
    L = (46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_tx) - a_hr +
         (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d) + C_m)
    
    return L


def log_distance_path_loss(distance_m: np.ndarray,
                          frequency_mhz: float,
                          n: float = 3.5,
                          d0: float = 1.0,
                          X_sigma: float = 0.0) -> np.ndarray:
    """Log-distance path loss: PL = FSPL(d0) + 10 n log10(d/d0) + X_sigma, in dB.

    d0 is the reference distance in metres (distances below d0 are clamped to d0), n the
    path-loss exponent and X_sigma a fixed shadowing offset in dB.
    """
    d = np.maximum(distance_m, d0)
    
    PL_d0 = free_space_path_loss(np.array([d0]), frequency_mhz)[0]
    
    PL = PL_d0 + 10 * n * np.log10(d / d0) + X_sigma
    
    return PL


class EmpiricalBaseline:
    """Wrapper that selects one of the empirical models by name and predicts received power."""
    
    MODELS = {
        'fspl': free_space_path_loss,
        'okumura_hata_urban': okumura_hata_urban,
        'okumura_hata_suburban': okumura_hata_suburban,
        'okumura_hata_rural': okumura_hata_rural,
        'cost231_urban': lambda d, f: cost231_hata(d, f, environment='urban'),
        'cost231_suburban': lambda d, f: cost231_hata(d, f, environment='suburban'),
        'log_distance': log_distance_path_loss
    }
    
    def __init__(self, model_name: str = 'fspl'):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(self.MODELS.keys())}")
        self.model_name = model_name
        self.model_fn = self.MODELS[model_name]
    
    def predict(self,
               distances: np.ndarray,
               frequency_mhz: float,
               p_tx_dbm: float = 30.0,
               **kwargs) -> Dict[str, np.ndarray]:
        """Return path loss (dB), received power p_tx_dbm - path loss (dBm) and a binary coverage
        flag (received power above -100 dBm).
        """
        path_loss = self.model_fn(distances, frequency_mhz, **kwargs)
        
        rssi = p_tx_dbm - path_loss
        
        coverage = (rssi > -100).astype(float)
        
        return {
            'path_loss_total': path_loss,
            'rssi': rssi,
            'coverage_prob': coverage
        }


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: Modelos Empíricos")
    print("=" * 60)
    
    distances = np.array([100, 500, 1000, 2000, 5000, 10000])
    frequency = 900.0
    
    print(f"\n📊 Frequência: {frequency} MHz")
    print(f"   Distâncias: {distances} m")
    
    print("\n📋 Path Loss (dB):")
    print("-" * 80)
    print(f"{'Distância':<12}", end="")
    for name in ['fspl', 'okumura_hata_urban', 'okumura_hata_suburban', 'cost231_urban']:
        print(f"{name:<20}", end="")
    print()
    print("-" * 80)
    
    for d in distances:
        print(f"{d:<12.0f}", end="")
        for name in ['fspl', 'okumura_hata_urban', 'okumura_hata_suburban', 'cost231_urban']:
            baseline = EmpiricalBaseline(name)
            result = baseline.predict(np.array([d]), frequency)
            print(f"{result['path_loss_total'][0]:<20.1f}", end="")
        print()
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO")
    print("=" * 60)
