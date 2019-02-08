from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans, KMeans, Birch, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import mixture
from scipy import interp
from tqdm import tqdm
import language_check
import numpy as np
import lda
import os


class SALT:
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
        :rtype: SALT
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
        """Train classifier. Refer to scikit-learn.org for documentation.
        :param kwargs: Arbitrary keyword arguments: classifier='SVM', kernel='poly', degree=2
                                                    classifier='SVM', kernel='sigmoid'
                                                    classifier='SVM', kernel='lin', gamma=2
                                                    classifier='KNN', k=3
                                                    classifier='DT', depth=5
                                                    classifier='RF', depth=5, n_estimators=10, max_features=1
                                                    classifier='NN', alpha=1, hidden_layer_sizes=(50,), max_iter=10,
                                                                     solver='sgd', ’adam’, activation=’relu’
                                                    classifier='AdaB'
                                                    classifier='GaussianNB'
                                                    classifier='MultinomialNB'
                                                    classifier='GP'
        :type kwargs: str, int
        :return: Object
        :rtype: SALT
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
                    else:
                        # kwargs['kernel'] is 'lin'
                        if 'gamma' in kwargs:
                            self.clf = SVC(gamma=kwargs['gamma'], C=1, probability=True)
                        else:
                            self.clf = SVC(gamma=2, C=1, probability=True)
                else:
                    # kwargs['kernel'] is 'lin'
                    if 'gamma' in kwargs:
                        self.clf = SVC(gamma=kwargs['gamma'], C=1, probability=True)
                    else:
                        self.clf = SVC(gamma=2, C=1, probability=True)
            elif kwargs['classifier'] == 'MultinomialNB':
                self.clf = MultinomialNB()
            elif kwargs['classifier'] == 'GaussianNB':
                self.clf = GaussianNB()
            elif kwargs['classifier'] == 'AdaB':
                self.clf = AdaBoostClassifier()
            elif kwargs['classifier'] == 'GP':
                self.clf = GaussianProcessClassifier(1.0 * RBF(1.0))
            elif kwargs['classifier'] == 'NN':
                if 'hidden_layer_sizes' in kwargs:
                    if 'max_iter' in kwargs:
                        if 'solver' in kwargs:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'])
                        else:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'],
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             max_iter=kwargs['max_iter'])
                    else:
                        if 'solver' in kwargs:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             solver=kwargs['solver'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             solver=kwargs['solver'])
                        else:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'],
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             hidden_layer_sizes=kwargs['hidden_layer_sizes'])
                else:
                    if 'max_iter' in kwargs:
                        if 'solver' in kwargs:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             max_iter=kwargs['max_iter'],
                                                             solver=kwargs['solver'])
                        else:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             max_iter=kwargs['max_iter'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             max_iter=kwargs['max_iter'],
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             max_iter=kwargs['max_iter'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             max_iter=kwargs['max_iter'])
                    else:
                        if 'solver' in kwargs:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             solver=kwargs['solver'],
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             solver=kwargs['solver'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             solver=kwargs['solver'])
                        else:
                            if 'activation' in kwargs:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'],
                                                             activation=kwargs['activation'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1,
                                                             activation=kwargs['activation'])
                            else:
                                if 'alpha' in kwargs:
                                    self.clf = MLPClassifier(alpha=kwargs['alpha'])
                                else:
                                    self.clf = MLPClassifier(alpha=0.1)

            elif kwargs['classifier'] == 'RF':
                if 'max_depth' in kwargs:
                    if 'n_estimators' in kwargs:
                        if 'max_features' in kwargs:
                            self.clf = RandomForestClassifier(max_depth=kwargs['max_depth'],
                                                              n_estimators=kwargs['n_estimators'],
                                                              max_features=kwargs['max_features'])
                        else:
                            self.clf = RandomForestClassifier(max_depth=kwargs['max_depth'],
                                                              n_estimators=kwargs['n_estimators'],
                                                              max_features=1)
                    else:
                        self.clf = RandomForestClassifier(max_depth=kwargs['max_depth'],
                                                          n_estimators=10,
                                                          max_features=1)
                else:
                    if 'n_estimators' in kwargs:
                        if 'max_features' in kwargs:
                            self.clf = RandomForestClassifier(max_depth=5,
                                                              n_estimators=kwargs['n_estimators'],
                                                              max_features=kwargs['max_features'])
                        else:
                            self.clf = RandomForestClassifier(max_depth=5,
                                                              n_estimators=kwargs['n_estimators'],
                                                              max_features=1)
                    else:
                        self.clf = RandomForestClassifier(max_depth=5,
                                                          n_estimators=10,
                                                          max_features=1)

            elif kwargs['classifier'] == 'KNN':
                if 'k' in kwargs:
                    self.clf = KNeighborsClassifier(kwargs['k'])
                else:
                    self. clf = KNeighborsClassifier(3)
            elif kwargs['classifier'] == 'DT':
                if 'depth' in kwargs:
                    self.clf = DecisionTreeClassifier(max_depth=kwargs['depth'])
                else:
                    self.clf = DecisionTreeClassifier(max_depth=5)
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
        x_test = vec.fit_transform(np.array(self.newdata))
        self.newdata = x_test.toarray()

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

    def enrich(self, method=None, num_clusters=2):
        if method == 'kmeans':
            self.kmeans_enrich(numclusters=num_clusters)
        elif method == 'mbk':
            self.mbk_enrich(numclusters=num_clusters)
        elif method == 'lda':
            self.lda_enrich(numclusters=num_clusters)
        elif method == 'birch':
            self.birch_enrich(numclusters=num_clusters)
        elif method == 'dbscan':
            self.dbscan_enrich()
        elif method == 'gmm':
            self.gmm_enrich(numclusters=num_clusters)
        elif method == 'ms':
            self.ms_enrich()
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
        labels = mbk.labels_
        cluster_centers = mbk.cluster_centers_
        n_features = self.vocabulary.__len__()
        for x in range(self.X.__len__()):
            gamma = np.count_nonzero(x) / n_features
            x_label = labels[x]
            center_vector = cluster_centers[x_label]
            for i in range(n_features):
                self.X[x][i] = self.X[x][i] + gamma * center_vector[i]

    def birch_enrich(self, numclusters=10, threshold=1.7):
        """Enrich the training set with BIRCH clustering algorithm.
        BIRCH (balanced iterative reducing and clustering using hierarchies) is an unsupervised data mining algorithm
        used to perform hierarchical clustering over particularly large data-sets. An advantage of BIRCH is its ability
        to incrementally and dynamically cluster incoming, multi-dimensional metric data points in an attempt to produce
        the best quality clustering for a given set of resources (memory and time constraints). In most cases, BIRCH
        only requires a single scan of the database.
        :param numclusters: Number of clusters
        :type numclusters: int
        :param threshold: The radius of the subcluster obtained by merging a new sample and the closest subcluster
        should be lesser than the threshold.
        :type threshold: float
        """
        birch = Birch(threshold=threshold, n_clusters=numclusters)
        birch.fit(self.X)
        labels = birch.labels_
        cluster_centers = birch.subcluster_centers_
        n_features = self.vocabulary.__len__()
        for x in range(self.X.__len__()):
            gamma = np.count_nonzero(x) / n_features
            x_label = labels[x]
            center_vector = cluster_centers[x_label]
            for i in range(n_features):
                self.X[x][i] = self.X[x][i] + gamma * center_vector[i]

    def gmm_enrich(self, numclusters=10):
        """Enrich the training set with Gaussian Mixture Modeling clustering algorithm.
        :param numclusters: Number of clusters
        :type numclusters: int
        """
        gmm = mixture.GaussianMixture(n_components=numclusters, covariance_type='full')
        gmm.fit(self.X)
        labels = gmm.predict(self.X)
        cluster_centers = gmm.means_
        n_features = self.vocabulary.__len__()
        for x in range(self.X.__len__()):
            gamma = np.count_nonzero(x) / n_features
            x_label = labels[x]
            center_vector = cluster_centers[x_label]
            for i in range(n_features):
                self.X[x][i] = self.X[x][i] + gamma * center_vector[i]

    def ms_enrich(self):
        """Enrich the training set with MeanShift clustering algorithm."""
        # bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(self.X, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(self.X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        # labels_unique = np.unique(labels)
        # n_clusters_ = len(labels_unique)
        n_features = self.vocabulary.__len__()
        for x in range(self.X.__len__()):
            gamma = np.count_nonzero(x) / n_features
            x_label = labels[x]
            center_vector = cluster_centers[x_label]
            for i in range(n_features):
                self.X[x][i] = self.X[x][i] + gamma * center_vector[i]

    def dbscan_enrich(self, eps=1.36):
        """Enrich the training set with DBSCAN clustering algorithm.
        Density-based spatial clustering of applications with noise (DBSCAN) is a well-known data clustering algorithm.
        Based on a set of points, DBSCAN groups together points that are close to each other based on a distance
        measurement (usually Euclidean distance) and a minimum number of points. It also marks as outliers the points
        that are in low-density regions. (DBSCAN does not compute the nearest core point or cluster center!)
        :param eps: the minimum distance between two points. It means that if the distance between two points is lower
        or equal to this value (eps), these points are considered neighbors.
        :type eps: float
        # minPoints: the minimum number of points to form a dense region. For example, if we set the minPoints parameter
         as 5, then we need at least 5 points to form a dense region.
        """
        dbscan = DBSCAN(eps)
        dbscan.fit(self.X)
        labels = dbscan.labels_
        # zeros_like returns an array of zeros with the same shape and type as a given array,
        # dtype will overrides the data type of the result.
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        # core_sample_indices_: index of core samples (array, shape = [n_core_samples])
        core_samples_mask[dbscan.core_sample_indices_] = True
        unique_labels = set(labels)
        # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_features = self.vocabulary.__len__()
        core_samples = []
        for k in unique_labels:
            class_member_mask = (labels == k)
            core_samples.append(self.X[class_member_mask & core_samples_mask])
        for x in range(self.X.__len__()):
            gamma = np.count_nonzero(x) / n_features
            x_label = labels[x]
            if x_label == -1:
                continue
            for i in range(n_features):
                self.X[x][i] = self.X[x][i] + gamma * (np.float64(core_samples[x_label][0:1][i]) +
                                                       np.float64(core_samples[x_label][1:2][i])) / 2

    def lda_enrich(self, numclusters=10):
        """Enrich the training set with LDA.
        :param numclusters: Number of clusters
        :type numclusters: int
        """
        # X = np.int32(self.X)
        self.X = np.array(self.X)
        model = lda.LDA(n_topics=numclusters, n_iter=1000, random_state=1)
        model.fit(self.X.astype(np.intp))
        topic_word = model.topic_word_
        # n_top_words = 8
        topics_distributions = []
        train_set = []
        n_features = self.vocabulary.__len__()
        for x in range(self.X.__len__()):
            if np.count_nonzero(self.X[x]) == 0:
                gamma = 0
            else:
                gamma = np.count_nonzero(self.X[x]) / n_features
            doc_topic_dist = model.doc_topic_[x]
            for i in range(n_features):
                lda_ev = 0
                for k, topic_dist in enumerate(topic_word):
                    lda_ev = lda_ev + doc_topic_dist[k] * topic_dist[i]
                self.X[x][i] = self.X[x][i] + gamma * lda_ev

            # topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            # topic_words_distributions = np.array(topic_dist)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            # print('Topic {}: {}'.format(i, ' '.join(topic_words)))
            # print('Distributions for Topic {}: {}'.format(i, topic_words_distributions))


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


def feature_selection():
    """
    training_feature_names = vec.get_feature_names()
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    selector = SelectKBest(mutual_info_classif, k=1000)
    X = selector.fit_transform(tfidf_matrix_train[n_topics:lenght_t], decursus_target)
    cols = selector.get_support(indices=True)
    training_feature_names = np.array(training_feature_names)[cols]
    """
