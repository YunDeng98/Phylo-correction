import os
import unittest
import tempfile
from parameterized import parameterized
import numpy as np
from filecmp import dircmp

from src.phyml.phyml import install_phyml


class TestPhyML(unittest.TestCase):
    def test_phyml(self):
        install_phyml()
