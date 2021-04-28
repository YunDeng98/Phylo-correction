import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class MSA:
    def __init__(
        self,
        a3m_dir: str,
        protein_family_name: str,
        max_seqs: int = 0,
        max_sites: int = 0,
    ) -> None:
        r"""
        Read an MSA.

        Reads the MSA from a3m_dir/protein_family_name.a3m.
        *Lowercase amino acids in the aligned sequences are ignored.* This is
        thus a form of MSA pre-processing that makes the MSA into a *valid*
        MSA (where all sequences have the same length) such that it can be
        consumed by FastTree. (FastTree errors out when run on the raw
        alignments!)
        If max_seqs is provided, max_seqs will be subsampled at random. The
        first sequence will always be kept (since it is the reference sequence).
        If max_sites is provided, max_sites positions will be subsampled at
        random.
        """
        filename = f"{protein_family_name}.a3m"
        if not os.path.exists(a3m_dir):
            raise ValueError(f"a3m_dir {a3m_dir} does not exist")
        filepath = os.path.join(a3m_dir, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"filepath {filepath} does not exist")

        # Read MSA
        msa = []  # type: List[Tuple[str, str]]
        with open(filepath) as file:
            lines = list(file)
            n_lines = len(lines)
            for i in range(0, n_lines, 2):
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

    def get_msa(self) -> Dict[str, str]:
        return self._msa_dict.copy()

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
        max_seqs: Optional[int],
        max_sites: Optional[int],
    ) -> List[Tuple[str, str]]:
        r"""
        Subsample an MSA.

        Subsamples max_seqs and max_sites from the MSA. Returns the new MSA.
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
        Writes the MSA to outfile.
        """
        with open(outfile, "w") as file:
            for protein_name, seq in self.msa:
                file.write(">" + protein_name + "\n")
                file.write(seq + "\n")
