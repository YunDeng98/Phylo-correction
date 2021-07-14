import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized

from src.transition_extraction import TransitionExtractor


class TestTransitionExtractor(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_basic_regression(self, name, n_process):
        """
        Test that TransitionExtractor runs and its output matches the expected output.
        The expected output is located at test_input_data/transitions_small

        We run the same TransitionExtractor twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'maximum_parsimony')
            for use_cached in [False, True]:
                transition_extractor = TransitionExtractor(
                    a3m_dir='test_input_data/a3m_small',
                    parsimony_dir='test_input_data/maximum_parsimony_small',
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    use_cached=use_cached,
                )
                transition_extractor.run()
                dcmp = dircmp(outdir, 'test_input_data/transitions_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)
            os.system(f"chmod -R 777 {root_dir}")
