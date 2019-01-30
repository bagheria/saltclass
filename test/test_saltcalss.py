import pytest
from saltclass import SALT


def test_is_instance(self):
    train_X = [[10, 0, 0], [0, 20, 0], [4, 13, 5]]
    train_y = [0, 1, 1]
    vocab = ['statistics', 'medicine', 'crime']
    assert(isinstance(SALT(train_X, train_y, vocabulary=vocab, language='en')))
