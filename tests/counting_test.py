import os
import unittest
import tempfile
from src.counting import JTT
import numpy as np


class TestJTT(unittest.TestCase):
    def test_smoke_toy_matrix(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset.
        """
        for ipw in [True, False]:
            with tempfile.TemporaryDirectory() as root_dir:
                outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
                for use_cached in [False, True]:
                    rate_matrix_learner = JTT(
                        frequency_matrices="test_input_data/matrices_toy.txt",
                        output_dir=outdir,
                        mask=None,
                        use_cached=use_cached,
                        ipw=ipw,
                    )
                    rate_matrix_learner.train()

                    ipw_str = "-IPW" if ipw else ""
                    expected_rate_matrix_path = f"test_input_data/Q1_JTT{ipw_str}_on_toy_matrix/learned_matrix.txt"
                    learned_rate_matrix_path = os.path.join(outdir, 'learned_matrix.txt')
                    np.testing.assert_almost_equal(
                        np.loadtxt(expected_rate_matrix_path),
                        np.loadtxt(learned_rate_matrix_path)
                    )

    def test_existing_results_are_not_overwritten(self):
        """
        We want to make sure we don't corrupt previous runs accidentaly.
        """
        for ipw in [True, False]:
            with tempfile.TemporaryDirectory() as root_dir:
                outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
                for i, use_cached in enumerate([False, False]):
                    rate_matrix_learner = JTT(
                        frequency_matrices="test_input_data/matrices_toy.txt",
                        output_dir=outdir,
                        mask=None,
                        use_cached=use_cached,
                        ipw=ipw,
                    )
                    if i == 0:
                        rate_matrix_learner.train()
                    else:
                        with self.assertRaises(PermissionError):
                            rate_matrix_learner.train()

    def test_smoke_large_matrix(self):
        """
        Test that JTT runs on a large input dataset.
        """
        for ipw in [True, False]:
            with tempfile.TemporaryDirectory() as root_dir:
                outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
                for use_cached in [False, True]:
                    rate_matrix_learner = JTT(
                        frequency_matrices="test_input_data/matrices_small/matrices_by_quantized_branch_length.txt",
                        output_dir=outdir,
                        mask=None,
                        use_cached=use_cached,
                        ipw=ipw,
                    )
                    rate_matrix_learner.train()

    def test_smoke_huge_matrix(self):
        """
        Test that JTT runs on a huge input dataset.
        """
        for ipw in [True, False]:
            with tempfile.TemporaryDirectory() as root_dir:
                outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
                for use_cached in [False, True]:
                    rate_matrix_learner = JTT(
                        frequency_matrices="test_input_data/co_matrices_small/matrices_by_quantized_branch_length.txt",
                        output_dir=outdir,
                        mask=None,
                        use_cached=use_cached,
                        ipw=ipw,
                    )
                    rate_matrix_learner.train()
