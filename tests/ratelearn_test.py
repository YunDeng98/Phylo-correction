import os
import unittest
import tempfile
# from filecmp import dircmp
# from parameterized import parameterized

from src.ratelearn import RateMatrixLearner


class TestRateMatrixLearner(unittest.TestCase):
    def test_smoke_toy_matrix(self):
        """
        Test that RateMatrixLearner runs.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'Q1_estimate')
            for use_cached in [False]:
                print("TODO: [False, True], and implement caching in ratelearn.")
                print("TODO: Test on large matrix! (the frequency_matrices commented out below)")
                rate_matrix_learner = RateMatrixLearner(
                    frequency_matrices="test_input_data/matrices_toy.txt",
                    # frequency_matrices="test_input_data/matrices_small/matrices_by_quantized_branch_length.txt",
                    output_dir=outdir,
                    stationnary_distribution=None,
                    mask=None,
                    # frequency_matrices_sep=",",
                    rate_matrix_parameterization="pande_reversible",
                )
                rate_matrix_learner.train(
                    lr=1e-1,
                    num_epochs=2000,
                    do_adam=True,
                )
