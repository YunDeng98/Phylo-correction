import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized
from src.counting import JTT


class TestJTT(unittest.TestCase):
    def test_smoke_toy_matrix(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
            for use_cached in [False, True]:
                rate_matrix_learner = JTT(
                    frequency_matrices="test_input_data/matrices_toy.txt",
                    output_dir=outdir,
                    mask=None,
                    use_cached=use_cached,
                )
                rate_matrix_learner.train()

    def test_existing_results_are_not_overwritten(self):
        """
        We want to make sure we don't corrupt previous runs accidentaly.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
            for i, use_cached in enumerate([False, False]):
                rate_matrix_learner = JTT(
                    frequency_matrices="test_input_data/matrices_toy.txt",
                    output_dir=outdir,
                    mask=None,
                    use_cached=use_cached,
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
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
            for use_cached in [False, True]:
                rate_matrix_learner = JTT(
                    frequency_matrices="test_input_data/matrices_small/matrices_by_quantized_branch_length.txt",
                    output_dir=outdir,
                    mask=None,
                    use_cached=use_cached,
                )
                rate_matrix_learner.train()

    def test_smoke_huge_matrix(self):
        """
        Test that JTT runs on a huge input dataset.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate_JTT')
            for use_cached in [False, True]:
                rate_matrix_learner = JTT(
                    frequency_matrices="test_input_data/co_matrices_small/matrices_by_quantized_branch_length.txt",
                    output_dir=outdir,
                    mask=None,
                    use_cached=use_cached,
                )
                rate_matrix_learner.train()
