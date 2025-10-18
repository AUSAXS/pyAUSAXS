class Molecule:
    def __init__(self, pdb_source):
        if isinstance(pdb_source, PDBFile):
            self.pdb_object_id = pdb_source.object_id
            self._pdb_file = pdb_source
        elif isinstance(pdb_source, int):
            self.pdb_object_id = pdb_source
            self._pdb_file = PDBFile.from_object_id(pdb_source)
        elif isinstance(pdb_source, str):
            self._pdb_file = PDBFile(pdb_source)
            self.pdb_object_id = self._pdb_file.object_id
        else:
            raise TypeError("pdb_source must be PDBFile, object ID (int), or filename (str)")

        # Create reduced representation for SAXS calculations
        self._create_reduced_representation()

    def _create_reduced_representation(self):
        """Create reduced molecular representation with coordinates and form factors."""
        # Get atomic data from PDB file
        x, y, z = self._pdb_file.get_coordinates()
        elements = self._pdb_file.get_elements()
        occupancies = self._pdb_file.get_occupancies()

        # TODO: Implement form factor calculation based on elements
        # This would use AUSAXS form factor libraries

        # For now, store basic information
        self.coordinates = np.column_stack((x, y, z))
        self.elements = elements
        self.occupancies = occupancies if len(occupancies) > 0 else np.ones(len(x))

        # Calculate effective weights (form factors * occupancy)
        # This is a simplified version - real implementation would use proper form factors
        self.weights = self.occupancies.copy()

    def get_coordinates(self) -> np.ndarray:
        """Get molecular coordinates as Nx3 numpy array."""
        return self.coordinates.copy()

    def get_atomic_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get coordinates as separate x, y, z arrays (compatible with AUSAXS functions)."""
        return self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]

    def get_weights(self) -> np.ndarray:
        """Get atomic weights/form factors."""
        return self.weights.copy()

    def get_atom_count(self) -> int:
        """Get number of atoms in the molecule."""
        return len(self.coordinates)

    def calculate_scattering(self, q_vector: Union[list[float], np.ndarray]) -> np.ndarray:
        ausaxs = AUSAXS()
        x, y, z = self.get_atomic_coordinates()
        return ausaxs.debye(q_vector, x, y, z, self.weights)


# Convenience functions for backward compatibility and easy access
def read_pdb(filename: str) -> PDBFile:
    """
    Read a PDB file and return a PDBFile object.
    
    Args:
        filename: Path to PDB or CIF file
        
    Returns:
        PDBFile object with object ID for efficient reuse
    """
    return PDBFile(filename)


def read_data(filename: str) -> DataFile:
    """
    Read a SAXS/SANS data file and return a DataFile object.
    
    Args:
        filename: Path to data file
        
    Returns:
        DataFile object with object ID for efficient reuse
    """
    return DataFile(filename)


def molecule_from_pdb(pdb_source) -> Molecule:
    """
    Create a Molecule object from PDB source.
    
    Args:
        pdb_source: PDBFile object, object ID, or filename
        
    Returns:
        Molecule object optimized for scattering calculations
    """
    return Molecule(pdb_source)