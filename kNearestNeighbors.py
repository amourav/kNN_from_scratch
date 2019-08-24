import numpy as np
from collections import Counter


def euc_dist(a, b):
    """
    calculate euclidean distance (l2 norm of a-b) between a and b
    :param a: data point X[i, :] (list / array) len(a) = n
    :param b: data point X[j, :] (list / array) len(b) = n
    :return: distance (float)
    """
    return np.linalg.norm(a - b, ord=2)


def accuracy(y_true, y_pred):
    """
    measure accuracy of predictions (y_pred) given true labels (y_true)
    :param y_true: true class labels (list / array) - e.g. [0, 1, 2, 1]
    :param y_pred: predicted class labels (list / array) e.g. [0, 2, 1, 1]
    :return: accuracy (float)
    """
    return sum(y_true == y_pred) / len(y_true)


def norm_data(X):
    """
    normalize data to have zero mean and unit variance
    :param X: input data (array) - X.shape = (n_samples, m_features)
    :return:
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, (mean, std)


def argsort(a):
    """
    sort list or array (ascending)
    :param a: list to be sorted
    :return: sorted list (array)
    """
    return np.array(a).argsort()


class kNearestNeighbor():
    """
    k nearest neighbour classifier - https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

    k - nearest neighbours (int) - default = 3
    dist_metric - distance metric (str) - default = euclidean
    norm - normalize data to zero mean
    and unit variance (bool) - default = True
    example:
    knn = kNearestNeighbor(k=3, dist_metric='euclidean', norm=True)
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)
    """

    def __init__(self, k=3, dist_metric='euclidean', norm=True):
        """
        :param k: nearest neighbours (int) - default = 3
        :param dist_metric: distance metric (str) - default = euclidean
        :param norm: normalize data to unit mean and variance (bool) - default = True
        """
        self.k = k
        self.isFit = False  # model fitting done?
        self.norm = norm
        self.dist_func = self.set_dist_func(dist_metric)

    def fit(self, X_train, y_train, v=False):
        """
        Define training data for
        :param X_train: training input data (array) - X.shape = (n_samples, m_features)
        :param y_train: training labels (array) - X.shape = (n_samples)
        :param v: verbose. print trn acc if True (bool)
        :return: None
        """
        # check data
        if self.norm:
            X_train, (trn_mean, trn_std) = norm_data(X_train)
            self.trn_mean = trn_mean
            self.trn_std = trn_std
        self.X_train = X_train
        self.y_train = y_train

        y_train_pred, y_train_pred_proba = [], []
        for i, x_i in enumerate(X_train):
            distances = []
            for j, x_j in enumerate(X_train):
                if i == j:
                    dist_ij = 0
                else:
                    dist_ij = self.dist_func(x_i, x_j)

                distances.append(dist_ij)
            pred_i = self.estimate_point(distances, y_train)
            y_train_pred_i, y_train_pred_proba_i = pred_i
            y_train_pred.append(y_train_pred_i)
            y_train_pred_proba.append(y_train_pred_proba_i)

        if v:
            trn_acc = accuracy(y_train, y_train_pred)
            print('training accuracy: {}'.format(trn_acc))

        self.isFit = True

    def estimate_point(self, distances, y):
        """
        estimate most likely class given k neighbours
        :param distances: distances to all other points (list)
        :param y: labels associated with each entry in distances
        :return: most likely class, probability of class
        """
        sort_idx = argsort(distances)
        y_closest = y[sort_idx][:self.k]
        most_common = Counter(y_closest).most_common(1)[0]
        y_pred_i = most_common[0]
        y_pred_proba_i = most_common[1] / len(y_closest)
        return y_pred_i, y_pred_proba_i

    def norm_new(self, X_new):
        """
        normalize test data based on mean and variance of training data
        :param X_new: input data of a new set of samples (array) -
        X_new.shape = (n_samples, m_features)
        :return: normalized data (array)
        """
        return (X_new - self.trn_mean) / self.trn_std

    def predict(self, X_new):
        """
        predict class labels based on training data
        :param X_new: input data of a new set of samples (array) -
        X_new.shape = (n_samples, m_features)
        :return: y_new_pred: predicted class labels of X_new (list)
        """
        if not (self.isFit):
            raise Exception('run knn.fit(x_trn, y_trn) before running knn.predict(x_new)')
        if self.norm:
            X_new = self.norm_new(X_new)

        y_new_pred, y_new_pred_proba = [], []
        for i, x_i in enumerate(X_new):
            distances = []
            for j, x_j in enumerate(self.X_train):
                dist_ij = self.dist_func(x_i, x_j)
                distances.append(dist_ij)

            pred_i = self.estimate_point(distances, self.y_train)
            y_pred_i, y_pred_proba_i = pred_i
            y_new_pred.append(y_pred_i)
            y_new_pred_proba.append(y_pred_proba_i)

        return y_new_pred

    def evaluate(self, y_true, y_pred):
        """
        evaluate
        :param X_new:
        :param y_new:
        :return:
        """
        pass

    def set_dist_func(self, dist_metric):
        """
        set distance metric
        :param dist_metric: method for measuring distance between points (str)
        default =  euclidean
        :return:
        """
        implemented_metrics = {'euclidean': euc_dist, }
        for metric in implemented_metrics.keys():
            if dist_metric == metric:
                return implemented_metrics[metric]
        raise Exception('{} is not an acceptable argument for dist_metric')
