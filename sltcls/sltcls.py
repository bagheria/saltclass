from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy import interp
from tqdm import tqdm
import language_check
import pandas as pd
import numpy as np
import os


class SLT:
    """Classify short/long text to the predefined categories using clustering-based enrichment."""
    def __init__(self, x, y, vocabulary, **kwargs):
        """Initialize the object with training matrix and setting.
        :param x: Data for training, a numerical document-term matrix
        :type x: list
        :param y: Target binary values
        :type y: list
        :param vocabulary: Name of variables in x
        :type vocabulary: list
        :param kwargs: Arbitrary keyword arguments: language='nl', vectorizer='count'
        :type kwargs: str
        """
        self.X = x   # train data
        self.y = y
        self.newdata = []  # test data
        self.prediction = 0
        if kwargs is not None and 'vectorizer' in kwargs:
            self.vectorizer = kwargs['vectorizer']
        else:
            self.vectorizer = 'count'

        if kwargs is not None and 'language' in kwargs:
            self.language = kwargs['language']
        else:
            self.language = 'nl'

        self.clf = None
        self.vocabulary = vocabulary

    @classmethod
    def data_from_dir(cls, **kwargs):
        """Initialize the object from a directory of text files.
        :param kwargs: Arbitrary keyword arguments: language='nl', vectorizer='count'
        :type kwargs: str
        :return: Object
        :rtype: SLT
        """
        if kwargs is not None and 'vectorizer' in kwargs:
            vectorizer = kwargs['vectorizer']
        else:
            vectorizer = 'count'

        if kwargs is not None and 'language' in kwargs:
            language = kwargs['language']
        else:
            language = 'nl'

        # train data
        x_train, y_train, vocabulary = initialize_dataset(train_path=kwargs['train_dir'], word_vectorizer=vectorizer,
                                                          language=language)
        stclassifier_object = cls(x_train, y_train, vocabulary, vectorizer=vectorizer, language=language)
        return stclassifier_object

    def train(self, **kwargs):
        """Train classifier.
        :param kwargs: Arbitrary keyword arguments: classifier='SVM', kernel='lin', degree=2, gamma=2
        :type kwargs: str, int
        :return: Object
        :rtype: SLT
        """
        if kwargs is not None and 'classifier' in kwargs:
            if kwargs['classifier'] == 'SVM':
                if 'kernel' in kwargs:
                    if kwargs['kernel'] == 'poly':
                        if 'degree' in kwargs:
                            self.clf = SVC(kernel='poly', degree=kwargs['degree'], C=1.0, probability=True)
                        else:
                            self.clf = SVC(kernel='poly', degree=2, C=1.0, probability=True)
                    elif kwargs['kernel'] == 'sigmoid':
                        self.clf = SVC(kernel='sigmoid', probability=True)
                else:  # kwargs['kernel'] is 'lin'
                    if 'gamma' in kwargs:
                        self.clf = SVC(gamma=kwargs['gamma'], C=1, probability=True)
                    else:
                        self.clf = SVC(gamma=2, C=1, probability=True)
        else:
            self.clf = MultinomialNB()

        self.clf.fit(self.X, self.y)

    def print_info(self):
        """Print object's information."""
        print("classifier=", self.clf)
        print("language=", self.language)
        print("vectorizer=", self.vectorizer)

    def test(self, test_dir, test_out_dir=None):
        """Predict categories of test data.
        :param test_dir: Directory for test data
        :type test_dir: str
        :param test_out_dir: Directory for output
        :type test_out_dir: str
        """
        for filename in os.listdir(test_dir):
            self.prediction = self.predict(test_dir + filename)
            print(filename, "-------->", self.prediction)

    def predict(self, data_file):
        """Predict category for a text file.
        :param data_file: A text file
        :type data_file: str
        :return: Prediction
        :rtype: str
        """
        f = open(data_file, "r")
        self.newdata = [f.read()]
        self.prepare_data()
        self.prediction = self.clf.predict(self.newdata)
        return self.prediction

    def prepare_data(self):
        """Transform the data with vectorizer."""
        if self.vectorizer == 'count':
            # can change to 'vocabulary=pickle.load(open("feature.pkl", "rb"))'
            vec = CountVectorizer(vocabulary=self.vocabulary)
        else:
            vec = TfidfVectorizer(vocabulary=self.vocabulary)

        x_test = vec.fit_transform(self.newdata)
        self.newdata = pd.DataFrame(x_test.toarray(), columns=self.vocabulary)

    # def predict(self, data_file, enrich=True):
    #     # enrich test data
    #     print()

    def binarize_y(self):
        """Make target binary."""
        # improve: can extract unique values of y to binarize based on
        value = self.y[0]
        binary_y = [1 if i == value else 0 for i in self.y]
        self.y = binary_y

    def plot_roc_curve(self):
        """Plot roc curves for cv."""
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        cv = StratifiedKFold(n_splits=5)

        i = 0
        for train, test in cv.split(self.X, self.y):
            probas_ = self.clf.fit(self.X.ix[train], [self.y[i] for i in train]).predict_proba(self.X.ix[test])
            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve([self.y[i] for i in test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.clf)  # improve: need the name
        plt.legend(loc="lower right")
        plt.show()

    def enrich(self, method=None, num_clusters=3):
        if method == 'kmeans':
            self.kmeans_enrich(numclusters=num_clusters)
        elif method == 'mbk':
            self.mbk_enrich(numclusters=num_clusters)
        else:
            self.kmeans_enrich(numclusters=num_clusters)

    def kmeans_enrich(self, numclusters=10):
        """Enrich the training set with kmeans algorithm.
        :param numclusters: Number of clusters
        :type numclusters: int
        """
        km = KMeans(n_clusters=numclusters, init='k-means++', max_iter=300, n_init=1, verbose=0, random_state=3425)
        km.fit(self.X)
        n_features = self.vocabulary.__len__()
        # feature_names = vec.get_feature_names() # self.vocabulary
        for x in range(self.X.__len__()):
            # check gamma, influence of length
            gamma = np.count_nonzero(x) / n_features
            x_label = km.labels_[x]
            center_vector = km.cluster_centers_[x_label]
            for i in range(n_features):  # (for each word in the cluster center)
                self.X[x][i] = self.X[x][i] + gamma * center_vector[i]

    def mbk_enrich(self, numclusters=10):
        """Enrich the training set with MiniBatchKMeans clustering algorithm.
        :param numclusters: Number of clusters
        :type numclusters: int
        """
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=numclusters, batch_size=100,
                              n_init=10, max_no_improvement=10, verbose=0,
                              random_state=0)
        mbk.fit(self.X)
        n_features = self.vocabulary.__len__()
        for x in range(self.X.__len__()):
            # check gamma, influence of length
            gamma = np.count_nonzero(x) / n_features
            x_label = mbk.labels_[x]
            center_vector = mbk.cluster_centers_[x_label]
            for i in range(n_features):  # (for each word in the cluster center)
                self.X[x][i] = self.X[x][i] + gamma * center_vector[i]

    def birch_enrich(self, numclusters=10, threshold=1.7):
        """Enrich the training set with MiniBatchKMeans clustering algorithm.
        BIRCH (balanced iterative reducing and clustering using hierarchies) is an unsupervised data mining algorithm
        used to perform hierarchical clustering over particularly large data-sets. An advantage of BIRCH is its ability
        to incrementally and dynamically cluster incoming, multi-dimensional metric data points in an attempt to produce
        the best quality clustering for a given set of resources (memory and time constraints). In most cases, BIRCH
        only requires a single scan of the database.
        :param numclusters: Number of clusters
        :type numclusters: int
        :param threshold: The radius of the subcluster obtained by merging a new sample and the closest subcluster
        should be lesser than the threshold.
        :type threshold: int
        """
        birch = Birch(threshold=threshold, n_clusters=numclusters)
        birch.fit(self.X)
        n_features = self.vocabulary.__len__()
        for x in range(self.X.__len__()):
            # check gamma, influence of length
            gamma = np.count_nonzero(x) / n_features
            x_label = birch.labels_[x]
            center_vector = birch.cluster_centers_[x_label]
            for i in range(n_features):  # (for each word in the cluster center)
                self.X[x][i] = self.X[x][i] + gamma * center_vector[i]


class Text:
    """Text as content and category."""
    def __init__(self, content, category):
        self.content = content
        self.category = category


def initialize_dataset(train_path, word_vectorizer, language='nl'):
    """Read documents from train data and vectorize the data.
    :param train_path: Directory for train data
    :type train_path: str
    :param word_vectorizer: Feature vectorizer
    :type word_vectorizer: str
    :param language: Language of text
    :type language: str
    :return: Transformed matrix, List of categories, and Vocabulary of data
    :rtype: list
    """
    categories = os.listdir(train_path)
    docs = []
    for i in range(categories.__len__()):
        path = train_path + str(categories[i]) + "/"
        print("Reading documents: ")
        for filename in tqdm(os.listdir(path)):
            f = open(path + filename, "r")
            docs.append(Text(f.read(), categories[i]))

    # add ngram
    # add grid_search
    if word_vectorizer == 'count':
        vec = CountVectorizer()
    else:
        vec = TfidfVectorizer()

    docs = spell_correction(docs, language)

    num_docs = docs.__len__()

    x = vec.fit_transform(np.array([docs[i].content for i in range(num_docs)]))
    x = x.toarray()
    categories = np.array([docs[i].category for i in range(num_docs)])

    vocab = vec.vocabulary_
    return x, categories, vocab


def spell_correction(X, language):
    """Spell correction for the selected language.
    :param X: Matrix of raw documents
    :type X: list
    :param language: Language of text
    :type language: str
    :return: Spell-corrected data
    :rtype: list
    """
    tool = language_check.LanguageTool(language)
    print("Spell correction: ")
    for i in tqdm(range(X.__len__())):
        matches = tool.check(X[i].content)
        X[i].content = language_check.correct(X[i].content, matches)

    return X


def detect_negation_scope(X, language):
    """Spell correction for the selected language.
    :param X: Matrix of raw documents
    :type X: list
    :param language: Language of text
    :type language: str
    :return:
    :rtype:
    """
