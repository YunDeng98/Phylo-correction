import os
import unittest
import tempfile
import numpy as np
# from filecmp import dircmp
# from parameterized import parameterized

from src.ratelearn import RateMatrixLearner


class TestRateMatrixLearner(unittest.TestCase):
    def test_smoke_toy_matrix(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate')
            for use_cached in [False, True]:
                rate_matrix_learner = RateMatrixLearner(
                    frequency_matrices="test_input_data/matrices_toy.txt",
                    output_dir=outdir,
                    stationnary_distribution=None,
                    mask=None,
                    # frequency_matrices_sep=",",
                    rate_matrix_parameterization="pande_reversible",
                    device='cpu',
                    use_cached=use_cached,
                    initialization=np.loadtxt("test_input_data/3x3_pande_reversible_initialization.txt"),
                )
                rate_matrix_learner.train(
                    lr=1e-1,
                    num_epochs=3,
                    do_adam=True,
                )

    def test_existing_results_are_not_overwritten(self):
        """
        We want to make sure we don't corrupt previous runs accidentaly.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate')
            for i, use_cached in enumerate([False, False]):
                rate_matrix_learner = RateMatrixLearner(
                    frequency_matrices="test_input_data/matrices_toy.txt",
                    output_dir=outdir,
                    stationnary_distribution=None,
                    mask=None,
                    # frequency_matrices_sep=",",
                    rate_matrix_parameterization="pande_reversible",
                    device='cpu',
                    use_cached=use_cached,
                )
                if i == 0:
                    rate_matrix_learner.train(
                        lr=1e-1,
                        num_epochs=3,
                        do_adam=True,
                    )
                else:
                    with self.assertRaises(PermissionError):
                        rate_matrix_learner.train(
                            lr=1e-1,
                            num_epochs=3,
                            do_adam=True,
                        )

    def test_smoke_large_matrix(self):
        """
        Test that RateMatrixLearner runs on a large input dataset.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate')
            for use_cached in [False, True]:
                rate_matrix_learner = RateMatrixLearner(
                    frequency_matrices="test_input_data/matrices_small/matrices_by_quantized_branch_length.txt",
                    output_dir=outdir,
                    stationnary_distribution=None,
                    mask=None,
                    # frequency_matrices_sep=",",
                    rate_matrix_parameterization="pande_reversible",
                    device='cpu',
                    use_cached=use_cached,
                )
                rate_matrix_learner.train(
                    lr=1e-1,
                    num_epochs=3,
                    do_adam=True,
                )

    def test_smoke_huge_matrix(self):
        """
        Test that RateMatrixLearner runs on a huge input dataset.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate')
            for use_cached in [False, True]:
                rate_matrix_learner = RateMatrixLearner(
                    frequency_matrices="test_input_data/co_matrices_small/matrices_by_quantized_branch_length.txt",
                    output_dir=outdir,
                    stationnary_distribution=None,
                    mask=None,
                    # frequency_matrices_sep=",",
                    rate_matrix_parameterization="pande_reversible",
                    device='cpu',
                    use_cached=use_cached,
                )
                rate_matrix_learner.train(
                    lr=1e-1,
                    num_epochs=1,
                    do_adam=True,
                )
