import os

import numpy as np

from biotite.structure.io.pdb import PDBFile
from scipy.spatial.distance import pdist, squareform


def extend(a, b, c, L, A, D) -> float:
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


class ContactMatrix:
    r"""
    Creates a contact matrix from a PDB file.

    Reads the PDB file at f'{pdb_dir}/{protein_family_name}.pdb' and
    computes the binary contact matrix based on the provided armstrong_cutoff.
    The contact matrix can be written out to a file with the write_to_file method.

    Args:
        pdb_dir: Directory where the pdb structure files (.pdb) are found.
        protein_family_name: Name of the protein family.
        armstrong_cutoff: Armstrong cutoff threshold used to determine if two
            sites are in contact.

    Attributes:
        nsites: Number of sites in the protein.
    """
    def __init__(
        self,
        pdb_dir: str,
        protein_family_name: str,
        armstrong_cutoff: float = 8.0,
    ):
        pdb_file = os.path.join(pdb_dir, protein_family_name + ".pdb")
        pdbfile = PDBFile.read(str(pdb_file))
        structure = pdbfile.get_structure()
        N = structure.coord[0, structure.atom_name == "N"]
        C = structure.coord[0, structure.atom_name == "C"]
        CA = structure.coord[0, structure.atom_name == "CA"]
        Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
        distogram = squareform(pdist(Cbeta))
        pdb_contact = 1 * (distogram < armstrong_cutoff)
        self._pdb_contact = pdb_contact

    @property
    def nsites(self) -> int:
        r"""
        Number of sites in the sequence
        """
        assert self._pdb_contact.shape[0] == self._pdb_contact.shape[1]
        return self._pdb_contact.shape[0]

    def write_to_file(self, outfile: str) -> None:
        r"""
        Writes the contact matrix to outfile. Spaces are used as separators.
        """
        res = ""
        n = self.nsites
        for i in range(n):
            for j in range(n):
                if j:
                    res += " "
                res += str(self._pdb_contact[i, j])
            res += "\n"
        with open(outfile, "w") as file:
            file.write(res)
            file.flush()
