import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized

from src.matrix_generation import MatrixGenerator


class TestMatrixGenerator(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_basic_regression_single_site(self, name, n_process):
        """
        Test that MatrixGenerator runs and its output matches the expected output.
        The expected output is located at test_input_data/matrices_small

        We run the same MatrixGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'maximum_parsimony')
            for use_cached in [False, True]:
                matrix_generator = MatrixGenerator(
                    a3m_dir='test_input_data/a3m_small',
                    transitions_dir='test_input_data/transitions_small',
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    num_sites=1,
                    use_cached=use_cached,
                )
                matrix_generator.run()
                dcmp = dircmp(outdir, 'test_input_data/matrices_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_basic_regression_pair_of_sites(self, name, n_process):
        """
        Test that MatrixGenerator runs and its output matches the expected output.
        The expected output is located at test_input_data/co_matrices_small

        We run the same MatrixGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'maximum_parsimony')
            for use_cached in [False, True]:
                matrix_generator = MatrixGenerator(
                    a3m_dir='test_input_data/a3m_small',
                    transitions_dir='test_input_data/co_transitions_small',
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    num_sites=2,
                    use_cached=use_cached,
                )
                matrix_generator.run()
                dcmp = dircmp(outdir, 'test_input_data/co_matrices_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)