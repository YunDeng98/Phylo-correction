import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized

from src.xrate.xrate_input_generation import XRATEInputGenerator
from src.xrate.xrate import XRATE, install_xrate


class TestXRATEInputGenerator(unittest.TestCase):
    def test_xrate_installation(self):
        install_xrate()

    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_xrate_input_generator(self, name, n_process):
        """
        Test that XRATEInputGenerator runs and its output matches the expected
        output. The expected output is located at test_input_data/stock

        We run the same XRATEInputGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'stockholm')
            for use_cached in [False, True]:
                xrate_input_generator = XRATEInputGenerator(
                    a3m_dir_full='test_input_data/a3m_small',
                    parsimony_dir='test_input_data/maximum_parsimony_small',
                    n_process=n_process,
                    expected_number_of_MSAs=3,
                    outdir=outdir,
                    max_families=3,
                    use_cached=use_cached,
                )
                xrate_input_generator.run()
                dcmp = dircmp(outdir, 'test_input_data/stockholm_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)

    # def test_xrate(self):
    #     """
    #     Test that XRATE runs and its output matches the expected
    #     output. The expected output is located at test_input_data/Q1_XRATE_small
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'xrate')
    #         for use_cached in [False, True]:
    #             xrate = XRATE(
    #                 a3m_dir_full='test_input_data/a3m_small',
    #                 xrate_input_dir='test_input_data/stockholm_small',
    #                 expected_number_of_MSAs=3,
    #                 outdir=outdir,
    #                 max_families=3,
    #                 use_cached=use_cached,
    #             )
    #             xrate.run()
    #             dcmp = dircmp(outdir, 'test_input_data/Q1_XRATE_small')
    #             diff_files = dcmp.diff_files
    #             assert(len(diff_files) == 0)
