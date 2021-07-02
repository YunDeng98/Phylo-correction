import os
import unittest
import pytest
from parameterized import parameterized
import tempfile
from filecmp import dircmp

from src.phylogeny_generation import PhylogenyGenerator


class TestPhylogenyGenerator(unittest.TestCase):
    def test_regression(self):
        """
        Test that PhylogenyGenerator runs and its output matches the expected output.

        We run the same PhylogenyGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'trees')
            for use_cached in [False, True]:
                phylogeny_generator = PhylogenyGenerator(
                    a3m_dir='test_input_data/a3m_small',
                    n_process=3,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_seqs=8,
                    max_sites=16,
                    max_families=3,
                    rate_matrix='None',
                    use_cached=use_cached,
                )
                phylogeny_generator.run()
                dcmp = dircmp(outdir, 'test_input_data/trees_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)
