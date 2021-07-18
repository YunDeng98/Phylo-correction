import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized

from src.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3)])  # Testing only multiprocess due to speed constraints.
    def test_basic_regression(self, name, n_process):
        """
        Test that Pipeline runs and its output matches the expected output.
        The expected output is located at test_input_data/matrices_small
        and test_input_data/co_matrices_small

        We run the same Pipeline twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'pipeline')
            for use_cached in [False, True]:
                pipeline = Pipeline(
                    outdir=outdir,
                    max_seqs=8,
                    max_sites=16,
                    armstrong_cutoff=8.0,
                    rate_matrix="None",
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    max_families=3,
                    a3m_dir="test_input_data/a3m_small",
                    pdb_dir="test_input_data/pdb_small",
                    use_cached=True,
                    num_epochs=1,
                    center=1.0,
                    step_size=0.1,
                    n_steps=0,
                    keep_outliers=True,
                    device='cpu',
                    precomputed_contact_dir=None,
                    precomputed_tree_dir=None,
                    precomputed_maximum_parsimony_dir=None,
                )
                pipeline.run()
                dcmp = dircmp(os.path.join(outdir, "matrices_8_seqs_16_sites__1.0_center_0.1_step_size_0_n_steps_True_outliers"), 'test_input_data/matrices_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)
                dcmp = dircmp(os.path.join(outdir, "co_matrices_8_seqs_16_sites_8.0__1.0_center_0.1_step_size_0_n_steps_True_outliers"), 'test_input_data/co_matrices_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)
            os.system(f"chmod -R 777 {root_dir}")
