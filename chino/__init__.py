# CHINO: CHanneling Instability Neural Operator
# A graph neural operator for reactive infiltration instability simulation.
#
# Architecture: Attention-augmented MeshGraphNet
# Physics: Darcy-Boussinesq with dissolution and injection source
# Reference: Sun et al. (2020), Frontiers in Earth Science

from .model import AttentionMeshGraphNet
from .loss import compute_losses, anomaly_l2
from .dataset import RealisationDataset, build_graph

__all__ = [
    "AttentionMeshGraphNet",
    "compute_losses",
    "anomaly_l2",
    "RealisationDataset",
    "build_graph",
]
