import pytest
import saltclass


def test_is_instance():
    train_X = [[10, 0, 0], [0, 20, 0], [4, 13, 5]]
    train_y = [0, 1, 1]
    vocab = ['statistics', 'medicine', 'crime']
    assert isinstance(saltclass.SALT(train_X, train_y, vocabulary=vocab, language='en'), saltclass.SALT)


def test_test_saltclass():
    salt = saltclass.SALT([[10, 0, 0], [0, 20, 0]], [0, 1], vocabulary=['statistics', 'medicine'])
    assert(salt.language == 'nl')
