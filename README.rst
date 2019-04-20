|Travis|_

.. |Travis| image:: https://travis-ci.com/bagheria/saltclass.svg?token=fYbdQUbpnoucyHyb3fs2&branch=master
.. _Travis: https://travis-ci.com/bagheria/saltclass

saltclass
---------

saltclass (Short and Long Text Classifier) is a Python module for text classification built under the MIT license.
The project was started in 2018 at the Department of Methodology & Statistics, Utrecht University.

Short text classification can be defined simply as follows: Given a set of documents with representation D and a set of labels C, define a function F that will assign a value from the set of C to each document in D. Since short text is characterized by shortness in the length, and sparsity in the representation, we try to optimize D and F in such a way that results in better performance in managing and analyzing EHR text data.

Figure below presents the semantic flowchart of the proposed intra-clustering method. In this framework, the clustering procedure is used as the heart of the approach, where it pumps cluster information throughout the body of text via the smoothing system, supplying text length and other information. This method is a hybrid technique, using benefits of different modules, including dictionary- and topic-based approaches, smoothing methods, and cluster information.

.. image:: https://github.com/bagheria/saltclass/blob/master/Architecture.png

Installation
------------

To install via pip::

    $ pip install saltclass
    $ pip install --upgrade saltclass


Methods overview
------------
.. image:: https://github.com/bagheria/saltclass/blob/master/methods.png

Sample Usage
------------

.. code:: python

Example 1:
    >>> import saltclass
    >>> train_X = [[10, 0, 0], [0, 20, 0], [4, 13, 5]]
    >>> train_y = [0, 1, 1]
    >>> vocab = ['statistics', 'medicine', 'crime']
    >>> object_from_df = saltclass.SALT(train_X, train_y, vocabulary=vocab, language='en')
    >>> X = [[10, 12, 0], [14, 3, 52]]
    >>> object_from_df.enrich(method='kmeans', include_unlabeled=True, unlabeled_matrix=X)
    >>> object_from_df.train(classifier='KNN', k=5)

Example 2:
    >>> object_from_df.enrich(method='lda', include_unlabeled=True, unlabeled_dir='/Data/unlabeled/')
    >>> object_from_df.train(classifier='nn', alpha=0.01, hidden_layer_sizes=(50,), max_iter=10, solver='adam', activation='relu')

Example 3:
    >>> object_from_file = SALT.data_from_dir(train_dir='/train/', language='nl')
    >>> object_from_df.enrich(include_unlabeled=False)
    >>> object_from_df.train(classifier='svm', kernel='sigmoid')
    >>> object_from_df.print_info()

    >>> prediction = object_from_df.predict(data_file='second_test.txt')
    >>> print(object_from_df.vocabulary)
    >>> print(object_from_df.newdata)
    >>> print([k for (k, v) in object_from_df.vocabulary.items() if object_from_df.newdata[0][v] != 0])
    >>> print(prediction)


Dependencies
~~~~~~~~~~~~

saltclass requires:

- Python (>= 3.5)
- NumPy (>= 1.11.0)
- SciPy (>= 0.17.0)
- LDA
- Scikit-learn (>= 0.20.0)
- Matplotlib (>= 3.0)
- Tqdm
- Language_check
