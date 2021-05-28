import os
import time
import tempfile

from MSA import MSA

import logging


class FastTreePhylogeny:
    def __init__(
        self,
        a3m_dir: str,
        protein_family_name: str,
        outdir: str,
        max_seqs: int = 0,
        max_sites: int = 0,
    ) -> None:
        r"""
        Run FastTree on a given MSA.

        Runs FastTree on the MSA at a3m_dir/protein_family_name.a3m and writes
        the output (a newick tree) to outdir/protein_family_name.newick.
        """
        if not os.path.exists(outdir):
            print(f"Creating outdir {outdir}")
            os.makedirs(outdir)

        # Read MSA into standardized format (lowercase amino acids are removed.)
        msa = MSA(
            a3m_dir, protein_family_name, max_seqs=max_seqs, max_sites=max_sites
        )
        # Write (standardized) MSA
        with tempfile.NamedTemporaryFile("w") as processed_msa_file:
            processed_msa_filename = processed_msa_file.name
            msa.write_to_file(processed_msa_filename)
            # Run FastTree on (standardized) MSA
            outfile = os.path.join(outdir, protein_family_name) + ".newick"
            logger = logging.getLogger("phylogeny_generation" + __name__)
            logger.debug(f"Running FastTree on MSA:\n{msa}")
            time_start = time.time()
            dir_path = os.path.dirname(os.path.realpath(__file__))
            os.system(f"{dir_path}/FastTree -quiet < {processed_msa_filename} > {outfile}")
            time_end = time.time()
            self._total_time = time_end - time_start
            logger.debug(f"Time taken: {self.total_time}")
            self._msa = msa
            self._protein_family_name = protein_family_name

    @property
    def total_time(self) -> float:
        r"""
        The total time it took to run FastTree.
        """
        return self._total_time

    @property
    def nseqs(self) -> int:
        r"""
        Number of sequences in the MSA.
        """
        return self._msa.nseqs

    @property
    def nsites(self) -> int:
        r"""
        Number of sites in the MSA.
        """
        return self._msa.nsites

    def __str__(self) -> str:
        res = (
            f"FastTreePhylogeny for {self._protein_family_name}\nTotal time: "
            f"{self.total_time} (s)\n"
        )
        res += str(self._msa)
        return res
