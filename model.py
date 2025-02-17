import numpy as np
from sklearn.neighbors import NearestNeighbors


class ElectricalMarkovChainClassifier():
    """
    Electrical Markov Chain Classifier.

    Now this class support only binary classification. Main method is
    based on connection between random walks and electrical networks.
    Search for probabilities is implemented using power iteration method.

    The author expresses gratitude to the authors of the article:
    https://www.cs.cmu.edu/~hovy/papers/13WWW-electrical-networks.pdf

    The idea of the classifier was taken from here, but the implementation
    is completely different.

    Parameters
    ----------
    graph_create_method : {'balls', 'knn'}, default='knn'
        -`'balls'`: a point connects to all
        its neighbors along a given radius;
        -`'knn'`: vertices are connected based on the method KNN,
        after normalizing the features.

    n_neighbors: int, default=None
        Neighbors parameter for the method KNN.

    radius: int, default=None
        Radius parameter for the method Neighbourhood.

    n_jobs: int, default=-1
        The number of parallel jobs to run for certain operations.
        If set to -1, the number of jobs is set to the number of CPU cores.
    """
    knn = 'knn'
    balls = 'balls'

    def __init__(
            self,
            graph_create_method: str = 'knn',
            n_neighbors: int = None,
            radius: int = None,
            n_jobs: int = -1,
    ):
        self.graph_create_method = graph_create_method
        self.transition_mat = None
        self.n_features = None
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_jobs = n_jobs
        self.pole_indices = None

    def fit(
            self,
            X: np.ndarray | list,
            y: np.ndarray | list
    ) -> 'ElectricalMarkovChainClassifier':
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {np.ndarray, list} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : np.ndarray of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = self._validate_data(X, y)
        self.pole_indices = np.where((y == -1) | (y == 1))[0]
        train_indices = np.concatenate([self.pole_indices, np.where(y == 0)[0]])
   
        self.y = y[train_indices]
        self.X = X[train_indices]

        self.n_pole = len(self.pole_indices)
        self.n_transition = len(train_indices) - len(self.pole_indices)

        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

        return self

    def predict_proba(
            self, X: np.ndarray | list,
            one_connection_mode: bool = False
        ) -> np.ndarray:
        """
        Predicts class labels for X.

        Parameters
        ----------
        X : {np.ndarray, list} of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        one_connection_mode: bool, default=False
            If the value is True, then the test vectors are connected one at a time.
            If there are no unlabeled objects in the training sample, then
            one_connection_mode=True will be the same as the regular KNN
            Attention! This mode is slow.

        Returns
        -------
        np.array of shape (n_samples, )
            Returns class labels probs
        """

        if one_connection_mode:
            y = np.array([])
            for ind in range(len(X)):
                tmp = self.predict_proba([X[ind]])
                y = np.concatenate((y, tmp))

        X, _ = self._validate_data(X, )
        transition_mat = None

        if self.graph_create_method == self.knn:
            transition_mat = self._knn_transition_mat(X)
        # else:
        #     transition_mat = self._balls_transition_mat(X)

        Q = transition_mat[self.n_pole:, self.n_pole:]
        R = transition_mat[self.n_pole:, :self.n_pole]
        N = np.linalg.inv(np.eye(X.shape[0] + self.n_transition) - Q)
        B = N @ R

        y = []

        for i in range(X.shape[0]):
            y.append(sum(B[self.n_transition + i][np.where(self.y == 1)[0]]))

        return np.array(y)

    @staticmethod
    def сalculate_k_neighbors_for_one(
        vectors: np.ndarray,
        vector: np.ndarray,
        k: int
    ) -> np.ndarray:
        """
        Calculate distances between a target vector and other vectors
        in a given array.

        Parameters
        ----------
        vectors : np.ndarray
            An array of vectors where each row represents a vector.

        vector : np.ndarray
            The target vector for which distances are calculated.

        k : int
            The number of smallest distances to return.

        Returns
        -------
        np.ndarray
            Array of indices representing k smallest distances between
            the target vector and other vectors.
        """
        distances = np.linalg.norm(vectors - vector, axis=1)

        return np.argsort(distances)[:k]

    def _knn_transition_mat(self, X) -> np.ndarray:
        """
        Create transition matrix for knn prediction.

        Parameters
        ----------
        X : {np.ndarray, list} of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        transition_mat : np.ndarray of shape (n_samples, n_samples)
            Transition matrix
        """
        n_samples_test = X.shape[0]
        n_samples = n_samples_test + self.n_pole + self.n_transition

        X_full = np.vstack((self.X, X))
        X_full = (X_full - self.mean) / np.sqrt(self.var)

        transition_mat = np.zeros((n_samples, n_samples))

        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            n_jobs=self.n_jobs
        )
        nn.fit(X_full)

        _, indices = nn.kneighbors(X_full)

        for j in range(n_samples):
            transition_mat[j][indices[j]] = 1

        np.fill_diagonal(transition_mat, 0)

        for i in range(n_samples):
            row_sum = sum(transition_mat[i])
            if row_sum != 0:
                transition_mat[i] = transition_mat[i] / row_sum

        return transition_mat

    def _validate_data(
            self,
            X: np.ndarray | list,
            y: np.ndarray | list = [],
            skip_check_y: bool = False
    ) -> tuple:
        """
        Validate parameters for fit, predict and predict_proba methods.

        Parameters
        ----------
        X : {np.ndarray, list} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : {np.ndarray, list} of shape (n_samples,), default=[]
            Target vector relative to X.

        skip_check_y : bool, default=False
            Flag for predict_proba and predict methods, where vector y
            is not needed.
        Returns
        -------
        X, y
            Tuple of data and target data, casted to numpy.ndarray type
            if necessary.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X type must be numpy.ndarray or list")
        if not (skip_check_y or isinstance(y, (np.ndarray, list))):
            raise TypeError("y type must be numpy.ndarray or list")

        if not (skip_check_y or isinstance(y, np.ndarray)):
            y = np.array(y)
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if not (skip_check_y or len(y.shape) == 1):
            raise ValueError("y must have one dimension")

        if not all(isinstance(X_j, (int, float)) for X_i in X for X_j in X_i):
            raise ValueError("all types of values in X must be float or int")

        return X, y
