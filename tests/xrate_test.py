import os
import unittest
import tempfile
from parameterized import parameterized
import numpy as np
from filecmp import dircmp

from src.xrate.xrate_input_generation import XRATEInputGenerator
from src.xrate.xrate import XRATE, install_xrate


class TestXRATEInputGenerator(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_xrate_input_generator(self, name, n_process):
        """
        Test that XRATEInputGenerator runs and its output matches the expected
        output. The expected output is located at
        test_input_data/stockholm_small

        We run the same XRATEInputGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        install_xrate()
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'stockholm_small')
            for use_cached in [False, True]:
                xrate_input_generator = XRATEInputGenerator(
                    a3m_dir_full='test_input_data/a3m_small',
                    parsimony_dir='test_input_data/maximum_parsimony_small',
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    use_site_specific_rates=False,
                    num_rate_categories=20,
                    use_cached=use_cached,
                )
                xrate_input_generator.run()
                dcmp = dircmp(outdir, 'test_input_data/stockholm_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_xrate_input_generator_with_site_rates(self, name, n_process):
        """
        Test that XRATEInputGenerator runs and its output matches the expected
        output. The expected output is located at
        test_input_data/stockholm_small_site_rates

        We run the same XRATEInputGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        install_xrate()
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'stockholm_small_site_rates')
            for use_cached in [False, True]:
                xrate_input_generator = XRATEInputGenerator(
                    a3m_dir_full='test_input_data/a3m_small',
                    parsimony_dir='test_input_data/maximum_parsimony_small',
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    use_site_specific_rates=True,
                    num_rate_categories=20,
                    use_cached=use_cached,
                )
                xrate_input_generator.run()
                dcmp = dircmp(outdir, 'test_input_data/stockholm_small_site_rates')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)

    def test_xrate(self):
        """
        Test that XRATE runs and its output matches the expected
        output. The expected output is located at test_input_data/Q1_XRATE_small
        """
        install_xrate()
        for xrate_grammar in [None, 'test_input_data/xrate_grammars/nullprot.eg']:
            with tempfile.TemporaryDirectory() as root_dir:
                outdir = os.path.join(root_dir, 'xrate')
                for use_cached in [False, True]:
                    xrate = XRATE(
                        a3m_dir_full='test_input_data/a3m_small',
                        xrate_input_dir='test_input_data/stockholm_small',
                        expected_number_of_MSAs=3,
                        outdir=outdir,
                        max_families=3,
                        xrate_grammar=xrate_grammar,
                        use_site_specific_rates=False,
                        num_rate_categories=20,
                        use_cached=use_cached,
                    )
                    xrate.run()
                    learned_matrix_true_path = "test_input_data/Q1_XRATE_small/learned_matrix.txt"
                    learned_matrix_inferred_path = os.path.join(outdir, "learned_matrix.txt")
                    Q_true = np.loadtxt(learned_matrix_true_path)
                    Q_inferred = np.loadtxt(learned_matrix_inferred_path)
                    l1_error = np.sum(np.abs(Q_true - Q_inferred))
                    assert(l1_error < 0.01)

                    learned_matrix_true_path = "test_input_data/Q1_XRATE_small/learned_matrix_normalized.txt"
                    learned_matrix_inferred_path = os.path.join(outdir, "learned_matrix_normalized.txt")
                    Q_true = np.loadtxt(learned_matrix_true_path)
                    Q_inferred = np.loadtxt(learned_matrix_inferred_path)
                    l1_error = np.sum(np.abs(Q_true - Q_inferred))
                    assert(l1_error < 0.01)

    def test_xrate_with_site_rates(self):
        """
        Test that XRATE runs with site rates and its output matches the expected
        output. The expected output is located at
        test_input_data/Q1_XRATE_small_site_rates
        """
        install_xrate()
        for xrate_grammar in [None, 'test_input_data/xrate_grammars/nullprot.eg']:
            with tempfile.TemporaryDirectory() as root_dir:
                outdir = os.path.join(root_dir, 'Q1_XRATE_small_site_rates')
                for use_cached in [False, True]:
                    xrate = XRATE(
                        a3m_dir_full='test_input_data/a3m_small',
                        xrate_input_dir='test_input_data/stockholm_small_site_rates',
                        expected_number_of_MSAs=3,
                        outdir=outdir,
                        max_families=3,
                        xrate_grammar=xrate_grammar,
                        use_site_specific_rates=True,
                        num_rate_categories=20,
                        use_cached=use_cached,
                    )
                    xrate.run()
                    learned_matrix_true_path = "test_input_data/Q1_XRATE_small_site_rates/learned_matrix.txt"
                    learned_matrix_inferred_path = os.path.join(outdir, "learned_matrix.txt")
                    Q_true = np.loadtxt(learned_matrix_true_path)
                    Q_inferred = np.loadtxt(learned_matrix_inferred_path)
                    l1_error = np.sum(np.abs(Q_true - Q_inferred))
                    assert(l1_error < 0.01)

                    learned_matrix_true_path = "test_input_data/Q1_XRATE_small_site_rates/learned_matrix_normalized.txt"
                    learned_matrix_inferred_path = os.path.join(outdir, "learned_matrix_normalized.txt")
                    Q_true = np.loadtxt(learned_matrix_true_path)
                    Q_inferred = np.loadtxt(learned_matrix_inferred_path)
                    l1_error = np.sum(np.abs(Q_true - Q_inferred))
                    assert(l1_error < 0.01)

    def test_xrate_with_WAG_initialization(self):
        """
        Test that XRATE runs initialized with WAG rate matrix.
        """
        install_xrate()
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'xrate')
            for use_cached in [False, True]:
                xrate = XRATE(
                    a3m_dir_full='test_input_data/a3m_small',
                    xrate_input_dir='test_input_data/stockholm_small',
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    xrate_grammar='test_input_data/WAG_matrix.txt',
                    use_site_specific_rates=False,
                    num_rate_categories=20,
                    use_cached=use_cached,
                )
                xrate.run()
