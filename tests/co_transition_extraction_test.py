import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized
from src.co_transition_extraction import CoTransitionExtractor


class TestCoTransitionExtractor(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_basic_regression(self, name, n_process):
        """
        Test that CoTransitionExtractor runs and its output matches the expected output.
        The expected output is located at test_input_data/co_transitions_small

        We run the same CoTransitionExtractor twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'co_transitions_small')
            for use_cached in [False, True]:
                co_transition_extractor = CoTransitionExtractor(
                    a3m_dir_full='test_input_data/a3m_small',
                    a3m_dir='test_input_data/a3m_small',
                    parsimony_dir='test_input_data/maximum_parsimony_small',
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    contact_dir='test_input_data/contacts_small',
                    edge_or_cherry="cherry",
                    use_cached=use_cached,
                )
                co_transition_extractor.run()
                dcmp = dircmp(outdir, 'test_input_data/co_transitions_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)
