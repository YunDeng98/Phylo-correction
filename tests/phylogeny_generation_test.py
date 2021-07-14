import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized

from src.phylogeny_generation import PhylogenyGenerator, PhylogenyGeneratorError, MSAError


class TestPhylogenyGenerator(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_basic_regression(self, name, n_process):
        """
        Test that PhylogenyGenerator runs and its output matches the expected output.
        The expected output is located at test_input_data/trees_small

        We run the same PhylogenyGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'trees')
            for use_cached in [False, True]:
                phylogeny_generator = PhylogenyGenerator(
                    a3m_dir='test_input_data/a3m_small',
                    n_process=n_process,
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
            os.system(f"chmod -R 777 {root_dir}")

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_custom_rate_matrix_runs_regression(self, name, n_process):
        """
        Tests the use of a custom rate matrix in FastTree.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'trees')
            phylogeny_generator = PhylogenyGenerator(
                a3m_dir='test_input_data/a3m_small',
                n_process=n_process,
                expected_number_of_MSAs=3,
                outdir=outdir,
                max_seqs=8,
                max_sites=16,
                max_families=3,
                rate_matrix='input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt',
                use_cached=False,
            )
            phylogeny_generator.run()
            dcmp = dircmp(outdir, 'test_input_data/trees_small_Q1_uniform')
            diff_files = dcmp.diff_files
            assert(len(diff_files) == 0)
            os.system(f"chmod -R 777 {root_dir}")

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_inexistent_rate_matrix_raises_error(self, name, n_process):
        """
        If the rate matrix passed to FastTree does not exist, we should error out.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'trees')
            phylogeny_generator = PhylogenyGenerator(
                a3m_dir='test_input_data/a3m_small',
                n_process=n_process,
                expected_number_of_MSAs=3,
                outdir=outdir,
                max_seqs=8,
                max_sites=16,
                max_families=3,
                rate_matrix='I-do-not-exist',
                use_cached=False,
            )
            with self.assertRaises(PhylogenyGeneratorError):
                phylogeny_generator.run()
            os.system(f"chmod -R 777 {root_dir}")

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_malformed_a3m_file_raises_error(self, name, n_process):
        """
        If the a3m data is corrupted, an error should be raised.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'trees')
            phylogeny_generator = PhylogenyGenerator(
                a3m_dir='test_input_data/a3m_small_corrupted',
                n_process=n_process,
                expected_number_of_MSAs=3,
                outdir=outdir,
                max_seqs=8,
                max_sites=16,
                max_families=3,
                rate_matrix='None',
                use_cached=False,
            )
            with self.assertRaises(MSAError):
                phylogeny_generator.run()
            os.system(f"chmod -R 777 {root_dir}")

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_incorrect_expected_number_of_MSAs_raises_error(self, name, n_process):
        """
        If the a3m directory has a different number of files from the
        expected number, an error should be raised.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'trees')
            phylogeny_generator = PhylogenyGenerator(
                a3m_dir='test_input_data/a3m_small',
                n_process=n_process,
                expected_number_of_MSAs=4,
                outdir=outdir,
                max_seqs=8,
                max_sites=16,
                max_families=3,
                rate_matrix='None',
                use_cached=False,
            )
            with self.assertRaises(MSAError):
                phylogeny_generator.run()
            os.system(f"chmod -R 777 {root_dir}")
