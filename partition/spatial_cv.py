"""Spatial block partition with a buffer, used to keep nearby nodes out of different partitions.

Nodes (planar coordinates in metres) are assigned to square blocks of side `grid_size_km`
(`SpatialKFold._assign_groups`). The study's trainers (training/frozen/train_*_c0_spatial.py,
training/modelo_v3/v3_common.py) use this block assignment and then shuffle the blocks with the
split seed and divide them into training, validation and test sets themselves. They call
`validate_spatial_cv` as a post-hoc check of the buffer and of disjointness.
`SpatialKFold.split` is the stand-alone k-fold version: blocks are shuffled with `random_state`
and dealt into `n_splits` folds. The held-out nodes closer than `buffer_km` to any training
node are then removed; the training set is not trimmed.

Usage: import as a module; `python spatial_cv.py` runs a self-test on 5000 random points in a
10 km x 10 km square (1 km blocks, 0.5 km buffer).
"""
import torch
import numpy as np
from typing import List, Tuple, Optional, Generator
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


class SpatialKFold:
    """K-fold split over square spatial blocks, with a buffer that trims the held-out fold.

    n_splits: number of folds; buffer_km: minimum distance from a held-out node to the training set;
    grid_size_km: block side; random_state: seed of the block shuffle (global numpy RNG).
    """

    def __init__(self,
                 n_splits: int = 5,
                 buffer_km: float = 2.0,
                 grid_size_km: float = 5.0,
                 random_state: int = 42):
        self.n_splits = n_splits
        self.buffer_km = buffer_km
        self.grid_size_km = grid_size_km
        self.random_state = random_state
    
    def split(self,
             positions: torch.Tensor
             ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_indices, val_indices) per fold; positions is [N, 2] in metres."""
        pos = positions.numpy() if isinstance(positions, torch.Tensor) else positions
        n_samples = len(pos)
        
        pos_km = pos / 1000.0  # coordinates are assumed to be in metres

        group_ids = self._assign_groups(pos_km)
        unique_groups = np.unique(group_ids)
        
        np.random.seed(self.random_state)
        np.random.shuffle(unique_groups)
        
        fold_size = len(unique_groups) // self.n_splits
        
        for fold in range(self.n_splits):
            start = fold * fold_size
            # The last fold also takes the remainder blocks.
            end = start + fold_size if fold < self.n_splits - 1 else len(unique_groups)
            val_groups = set(unique_groups[start:end])
            
            val_mask = np.isin(group_ids, list(val_groups))
            train_mask = ~val_mask
            
            if self.buffer_km > 0:
                train_mask, val_mask = self._apply_buffer(
                    pos_km, train_mask, val_mask
                )
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            yield train_idx, val_idx
    
    def _assign_groups(self, positions_km: np.ndarray) -> np.ndarray:
        """Block id per node: integer grid indices (truncation of coordinate / block side), flattened row-major."""
        grid_x = (positions_km[:, 0] / self.grid_size_km).astype(int)
        grid_y = (positions_km[:, 1] / self.grid_size_km).astype(int)
        
        max_y = grid_y.max() + 1
        group_ids = grid_x * max_y + grid_y
        
        return group_ids
    
    def _apply_buffer(self,
                     positions_km: np.ndarray,
                     train_mask: np.ndarray,
                     val_mask: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Drop held-out nodes whose nearest training node is closer than buffer_km (exact cKDTree query)."""
        from scipy.spatial import cKDTree
        
        train_pos = positions_km[train_mask]
        val_pos = positions_km[val_mask]
        
        if len(train_pos) == 0 or len(val_pos) == 0:
            return train_mask, val_mask
        
        train_tree = cKDTree(train_pos)
        
        distances, _ = train_tree.query(val_pos, k=1)
        
        val_idx_original = np.where(val_mask)[0]
        too_close = distances < self.buffer_km
        
        new_val_mask = val_mask.copy()
        new_val_mask[val_idx_original[too_close]] = False
        
        return train_mask, new_val_mask


def validate_spatial_cv(train_idx: np.ndarray,
                       val_idx: np.ndarray,
                       positions: np.ndarray,
                       buffer_km: float = 2.0,
                       verbose: bool = True) -> bool:
    """Check that the two index sets are disjoint and at least buffer_km apart (positions in metres).

    The distance check uses up to 1000 held-out nodes drawn with the global numpy RNG (not seeded here).
    """
    from scipy.spatial.distance import cdist
    
    errors = []
    
    train_pos = positions[train_idx] / 1000.0
    val_pos = positions[val_idx] / 1000.0
    
    if len(train_pos) > 0 and len(val_pos) > 0:
        sample_size = min(1000, len(val_pos))
        sample_idx = np.random.choice(len(val_pos), sample_size, replace=False)
        
        dists = cdist(val_pos[sample_idx], train_pos)
        min_dist = dists.min()
        
        if min_dist < buffer_km:
            errors.append(f"Buffer violated: min_dist={min_dist:.2f}km < {buffer_km}km")
    
    overlap = set(train_idx) & set(val_idx)
    if len(overlap) > 0:
        errors.append(f"Overlap between train and val: {len(overlap)} samples")
    
    if errors:
        for e in errors:
            print(f"❌ VALIDATE Error: {e}")
        return False
    
    if verbose:
        print(f"✅ validate_spatial_cv PASSED")
        print(f"   Train samples: {len(train_idx)}")
        print(f"   Val samples: {len(val_idx)}")
        if len(train_pos) > 0 and len(val_pos) > 0:
            print(f"   Min distance: {min_dist:.2f} km")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: SpatialKFold")
    print("=" * 60)
    
    n = 5000
    positions = torch.rand(n, 2) * 10000
    
    splitter = SpatialKFold(
        n_splits=5,
        buffer_km=0.5,
        grid_size_km=1.0
    )
    
    print(f"\n📊 Testando {splitter.n_splits} folds:")
    
    all_valid = True
    for fold, (train_idx, val_idx) in enumerate(splitter.split(positions)):
        print(f"\n   Fold {fold + 1}:")
        valid = validate_spatial_cv(
            train_idx, val_idx, 
            positions.numpy(),
            buffer_km=splitter.buffer_km,
            verbose=False
        )
        
        if valid:
            print(f"      ✅ Train: {len(train_idx)}, Val: {len(val_idx)}")
        else:
            print(f"      ❌ Validation failed")
            all_valid = False
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if all_valid else "❌ TESTE FALHOU")
    print("=" * 60)
