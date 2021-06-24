import os
from typing import Dict, List, Tuple

import numpy as np


class MSA:
    r"""
    An MSA contains a multiple sequence alignment of proteins.

    The MSA is read from the MSA file f'{a3m_dir}/{protein_family_name}.a3m'.
    *Lowercase amino acids in the MSA file are ignored.*
    After ignoring all lowercase letters, all sequences in the MSA should have
    the same length. The main purpose of this class is to enable reading and
    preprocessing of MSA files. By preprocessing we specifically mean removing all
    lowercase letters from the alignment. The preprocessed MSA can be written out
    to a file with the write_to_file method. The MSA can be obtained as a dictionary
    mapping protein name to sequence with the get_msa method. The number of sequences
    and sites after preprocessing can be obtained via the nseqs and nsites attributes.
    To get the sequence of a specific protein, one can use the get_sequence method.

    Args:
        a3m_dir: Directory where the MSA file is found.
        protein_family_name: Name of the protein family.
        max_seqs: If nonzero, this number of sequences in the MSA file will be subsampled
            uniformly at random. The first sequence in the MSA file will always be sampled.
        max_sites: If nonzero, this number of sites in the MSA file will be subsampled
            uniformly at random.

    Attributes:
        nseqs: Number of sequences in the MSA (after subsamping)
        nsites: Number of sites in the MSA (after subsampling)
    """
    def __init__(
        self,
        a3m_dir: str,
        protein_family_name: str,
        max_seqs: int,
        max_sites: int,
    ) -> None:
        filename = f"{protein_family_name}.a3m"
        if not os.path.exists(a3m_dir):
            raise ValueError(f"a3m_dir {a3m_dir} does not exist!")
        filepath = os.path.join(a3m_dir, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"MSA file {filepath} does not exist!")

        # Read MSA
        msa = []  # type: List[Tuple[str, str]]
        with open(filepath) as file:
            lines = list(file)
            n_lines = len(lines)
            for i in range(0, n_lines, 2):
                if not lines[i][0] == '>':
                    raise ValueError("Protein name line should start with '>'")
                protein_name = lines[i][1:].strip()
                protein_seq = lines[i + 1].strip()
                # Lowercase amino acids in the sequence are repetitive
                # sequences and should be ignored.
                protein_seq = "".join(
                    [c for c in protein_seq if not c.islower()]
                )
                msa.append((protein_name, protein_seq))
            # Check that all sequences in the MSA have the same length.
            for i in range(len(msa) - 1):
                if len(msa[i][1]) != len(msa[i + 1][1]):
                    raise ValueError(
                        f"Sequence\n{msa[i][1]}\nand\n{msa[i + 1][1]}\nin the "
                        f"MSA do not have the same length! ({len(msa[i][1])} vs"
                        f" {len(msa[i + 1][1])})"
                    )
            msa = MSA._subsample_msa(msa, max_seqs, max_sites)
            self.msa = msa
            self.protein_family_name = protein_family_name
            self._nseqs = len(msa)
            self._nsites = len(msa[0][1])
            self._msa_dict = dict(msa)

    def get_sequence(self, sequence_name: str) -> str:
        r"""
        Returns the sequence corresponding to 'sequence_name'
        """
        return self._msa_dict[sequence_name]

    def get_msa(self, copy: bool) -> Dict[str, str]:
        r"""
        Return the MSA as a dictionary.

        The MSA is a dictionary that maps protein names to sequence.
        A reference to the internal structure of the MSA class is returned
        by if 'copy=False' - breaking encapsulation - so use with care.
        To return a copy, use 'copy=True'.
        """
        if copy:
            return self._msa_dict.copy()
        else:
            return self._msa_dict

    @property
    def nseqs(self) -> int:
        r"""
        Number of sequences in the MSA.
        """
        return self._nseqs

    @property
    def nsites(self) -> int:
        r"""
        Number of sites in the MSA.
        """
        return self._nsites

    @staticmethod
    def _subsample_msa(
        msa: List[Tuple[str, str]],
        max_seqs: int,
        max_sites: int,
    ) -> List[Tuple[str, str]]:
        r"""
        Subsample an MSA (not in-place).

        Subsamples max_seqs sequences and max_sites sites from the MSA.
        Returns a new MSA. The first sequence in the MSA is always kept.
        If max_seqs is 0 or None, all sequences will be kept.
        If max_sites is 0 or None, all sites will be kept.
        If max_seqs is greater than the number of sequences, all sequences will be kept.
        If max_sites is greater than the number of sites, all sites will be kept.
        """
        nseqs = len(msa)
        nsites = len(msa[0][1])
        if max_seqs:
            max_seqs = min(nseqs, max_seqs)
            seqs_to_keep = [0] + list(
                np.random.choice(
                    range(1, nseqs, 1), size=max_seqs - 1, replace=False
                )
            )
            seqs_to_keep = sorted(seqs_to_keep)
            msa = [msa[i] for i in seqs_to_keep]
        if max_sites:
            max_sites = min(nsites, max_sites)
            sites_to_keep = np.random.choice(
                range(nsites), size=max_sites, replace=False
            )
            sites_to_keep = sorted(sites_to_keep)
            msa = [
                (msa[i][0], MSA._subsample_sites(msa[i][1], sites_to_keep))
                for i in range(len(msa))
            ]
        return msa

    @staticmethod
    def _subsample_sites(seq: str, sites_to_keep: List[int]) -> str:
        return "".join([seq[i] for i in sites_to_keep])

    def __str__(self) -> str:
        res = (
            f"MSA\nProtein Family Name: {self.protein_family_name}\nNum"
            f" sequences = {self.nseqs}\nNum sites = {self.nsites}"
        )
        return res

    def write_to_file(self, outfile: str) -> None:
        r"""
        Writes the MSA to outfile in the a3m format.
        """
        with open(outfile, "w") as file:
            for protein_name, seq in self.msa:
                file.write(">" + protein_name + "\n")
                file.write(seq + "\n")
