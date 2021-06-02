import os
import time
import tempfile

from .MSA import MSA

import logging


class FastTreePhylogeny:
    def __init__(
        self,
        a3m_dir: str,
        protein_family_name: str,
        outdir: str,
        max_seqs: int = 0,
        max_sites: int = 0,
        rate_matrix: str = '',
        use_cached: bool = False,
    ) -> None:
        r"""
        Run FastTree on a given MSA.

        Runs FastTree on the MSA at a3m_dir/protein_family_name.a3m and writes
        the output (a newick tree) to outdir/protein_family_name.newick.

        Uses the rate matrix at 'rate_matrix'. If 'rate_matrix' ends in 'None',
        the default FastTree rate matrix is used.
        """
        logger = logging.getLogger("FastTreePhylogeny")

        # Caching pattern: skip any computation as soon as possible
        outfile = os.path.join(outdir, protein_family_name) + ".newick"
        if use_cached and os.path.exists(outfile):
            logger.info(f"Skipping. Cached FastTree output for family {protein_family_name} at {outfile}")
            return

        dir_path = os.path.dirname(os.path.realpath(__file__))
        fast_tree_bin_path = os.path.join(dir_path, 'FastTree')
        fast_tree_path = os.path.join(dir_path, 'FastTree.c')
        if not os.path.exists(fast_tree_bin_path):
            logger.info("Getting FastTree.c file ...")
            os.system(
                f"wget http://www.microbesonline.org/fasttree/FastTree.c -P {dir_path}"
            )
            logger.info("Compiling FastTree ...")
            # See http://www.microbesonline.org/fasttree/#Install
            os.system(
                f"gcc -DNO_SSE -O3 -finline-functions -funroll-loops -Wall -o {fast_tree_bin_path} {fast_tree_path} -lm"
            )

        if not os.path.exists(outdir):
            logger.info(f"Creating outdir {outdir}")
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
            time_start = time.time()
            dir_path = os.path.dirname(os.path.realpath(__file__))
            if rate_matrix[-4:] == 'None':
                assert(not os.path.exists(rate_matrix))
                logger.info(f"Running FastTree with default rate matrix on MSA:\n{msa}")
                os.system(f"{dir_path}/FastTree -quiet < {processed_msa_filename} > {outfile}")
            else:
                if not os.path.exists(rate_matrix):
                    logger.error(f"Could not find rate matrix {rate_matrix}")
                    exit(1)
                logger.info(f"Running FastTree with rate matrix {rate_matrix} on MSA:\n{msa}")
                os.system(f"{dir_path}/FastTree -quiet -trans {rate_matrix} < {processed_msa_filename} > {outfile}")
            time_end = time.time()
            self._total_time = time_end - time_start
            logger.info(f"Time taken: {self.total_time}")
            self._msa = msa
            self._protein_family_name = protein_family_name

            os.system(f"chmod -R 555 {outfile}")

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
