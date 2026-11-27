from .Molecule import Molecule, PROFILE, PROFILE_USERQ, debye_profile
from pyausaxs.signatures import register
import numpy as np

register({
    "molecule_debye_exact":       PROFILE,
    "molecule_debye_exact_userq": PROFILE_USERQ,
})

class unoptimized():
    @staticmethod
    def debye_exact(molecule: Molecule, q_vals: list[float] | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the exact Debye scattering intensity of the molecule. No form factors will be applied. 
        Warning: This method is _not_ optimized, and may be very slow for large molecules. It is only meant for testing and validation purposes.
        Returns: (q, I)
        """
        return debye_profile("molecule_debye_exact", molecule._get_id(), q_vals)
