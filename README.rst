|Travis|_

.. |Travis| image:: https://travis-ci.com/bagheria/saltclass.svg?token=fYbdQUbpnoucyHyb3fs2&branch=master
.. _Travis: https://travis-ci.com/bagheria/saltclass

saltclass
---------

saltclass (Short and Long Text Classifier) is a Python module for text classification built under the MIT license.
The project was started in 2018 at the Department of Methodology & Statistics, Utrecht University.


Installation
------------

To install via pip::

    $ pip install saltclass
    $ pip install --upgrade saltclass

Sample Usage
````````````
.. code:: python

    >>> from saltclass import SALT

Define object from dataframe:
    >>> train_X = [[10, 0, 0], [0, 20, 0], [4, 13, 5]]
    >>> train_y = [0, 1, 1]
    >>> vocab = ['statistics', 'medicine', 'crime']
    >>> object_from_df = SALT(train_X, train_y, vocabulary=vocab)

Define object from file directory:
    >>> object_from_file = SALT.data_from_dir(train_dir='D:/train/', language='en')

    >>> object_from_df.enrich()
    >>> object_from_df.train(classifier='svm')
    >>> object_from_df.print_info()

    >>> prediction = object_from_df.predict(data_file='second_test.txt')
    >>> print(object_from_df.vocabulary)
    >>> print(object_from_df.newdata)
    >>> print([k for (k, v) in object_from_df.vocabulary.items() if object_from_df.newdata[0][v] != 0])
    >>> print(prediction)
