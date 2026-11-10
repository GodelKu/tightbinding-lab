"""tblab: a from-scratch tight-binding teaching toolkit verified against analytic results."""

from .analysis import dirac_velocity, find_dirac_points
from .bands import BandResult, compute_bands
from .dos import dos_gaussian, dos_tetrahedron
from .kronig_penney import KronigPenney
from .models import SSH, Chain, Graphene, Haldane, Square, make_model
from .ssh import ssh_edge_modes, ssh_open_hamiltonian
from .topology import chern_number, zak_phase

__version__ = "0.1.0"

__all__ = [
    "SSH",
    "BandResult",
    "Chain",
    "Graphene",
    "Haldane",
    "KronigPenney",
    "Square",
    "__version__",
    "chern_number",
    "compute_bands",
    "dirac_velocity",
    "dos_gaussian",
    "dos_tetrahedron",
    "find_dirac_points",
    "make_model",
    "ssh_edge_modes",
    "ssh_open_hamiltonian",
    "zak_phase",
]
