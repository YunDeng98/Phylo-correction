import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized

from src.contact_generation import ContactGenerator


class TestContactGenerator(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    def test_basic_regression(self, name, n_process):
        """
        Test that ContactGenerator runs and its output matches the expected output.
        The expected output is located at test_input_data/contacts_small

        We run the same ContactGenerator twice: first without caching,
        then with caching, to make sure that caching works.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'contacts')
            for use_cached in [False, True]:
                contact_generator = ContactGenerator(
                    a3m_dir='test_input_data/a3m_small',
                    pdb_dir='test_input_data/pdb_small',
                    armstrong_cutoff=8.0,
                    n_process=n_process,
                    expected_number_of_families=3,
                    outdir=outdir,
                    max_families=3,
                    use_cached=use_cached,
                )
                contact_generator.run()
                dcmp = dircmp(outdir, 'test_input_data/contacts_small')
                diff_files = dcmp.diff_files
                assert(len(diff_files) == 0)
            os.system(f"chmod -R 777 {root_dir}")
