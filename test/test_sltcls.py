import unittest
import os
import shutil

from sltcls import SLT

class TestSLT(unittest.TestCase):
    def test_is_string(self):
        train_X = [[10, 0, 0], [0, 20, 0], [4, 13, 5]]
        train_y = [0, 1, 1]
        vocabulary = ['statistics', 'medicine', 'crime']
        s = SLT(train_X, train_y, vocabulary=vocabulary, language='en')
        self.assertTrue(isinstance(s, ))

