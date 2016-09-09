#!user/bin/ env python
# Author: Peng Hao
# Email: peng.hao@student.uts.edu.au


from math import sqrt
from random import randint
import numpy as np


class CBT(object):
    """
    This code is used to implement Code Book Transfer (CBT) model
    proposed in paper:
    Bin Li, Qiang Yang and Xiangyang Xue, IJCAI, 2009.
    'Can Movies and Books Collaborate?
    Cross-Domain Collaborative filtering for Sparsity Reduction'
    """

    def __init__(self, tar_train_mat, tar_test_mat, src_mat, user_cluster, item_cluster,
                 n_iter=100,
                 n_neighbor=20):
        """
        Parameters:
        ---------------
        tar_train_mat : scipy.csr_matrix
            A target domain train sparse rating matrix for users (rows) and items (cols).

        tar_test_mat : scipy.csr_matrix
            A target domain test sparse rating matrix for users (rows) and items (cols).

        src_mat : 2 * D numpy.ndarray
            A source domain dense rating matrix for users (rows) and items (cols).

        user_cluster : int
            The number of user clusters.

        item_cluster : int
            The number of item clusters.

        n_iter : int, default 100.
            The maximum iteration times.

        n_neighbor : int, default 20
            The number of neighbors for PCC measure.
        """
        self.tar_train_mat = tar_train_mat
        self.tar_test_mat = tar_test_mat
        self.src_mat = src_mat
        self.user_cluster = user_cluster
        self.item_cluster = item_cluster
        self.n_iter = n_iter
        self.n_neighbor = n_neighbor
        # Initialize user- and item-group matrices
        self.src_user_gp = np.random.rand(src_mat.shape[0], user_cluster)
        self.src_item_gp = np.random.rand(src_mat.shape[1], item_cluster)
        self.tar_user_gp = np.zeros((tar_train_mat.shape[0], user_cluster))
        self.tar_item_gp = np.zeros((tar_train_mat.shape[1], item_cluster))
        # Initialize source domain cluster-level rating pattern
        self.pattern = np.random.random((user_cluster, item_cluster))
        # Set indicator matrix for target train matrix
        self.indicator = tar_train_mat.sign()

    def fit(self):
        """
        Fit the CBT model then make prediction.
        """
        # Code book construction
        self.cb_construct()
        # Code book transfer
        self.cb_transfer()
        # Call KNN with PCC similarity measure to predict missing ratings.
        mae, rmse = self.knn_predict()
        return mae, rmse

    def cb_construct(self):
        """
        Code Book Construction in the source domain.
        """
        for run in xrange(self.n_iter):
            # Updating with Non-negative Matrix Tri-Factorization.
            # Update source user-group matrix
            XGST = self.src_mat.dot(self.src_item_gp).dot(self.pattern.T)
            FFTXGST = (self.src_user_gp.dot(self.src_user_gp.T)).dot(XGST)
            self.src_user_gp *= XGST / FFTXGST
            # Update source item_group matrix
            XTFS = (self.src_mat.T.dot(self.src_user_gp)).dot(self.pattern)
            GGTXTFS = self.src_item_gp.dot(self.src_item_gp.T).dot(XTFS)
            self.src_item_gp *= XTFS / GGTXTFS
            # Update source code book
            FTXG = (self.src_user_gp.T.dot(self.src_mat)).dot(self.src_item_gp)
            FTFSGTG = (self.src_user_gp.T.dot(self.src_user_gp)).dot(self.pattern).dot(
                    self.src_item_gp.T.dot(self.src_item_gp))
            self.pattern *= FTXG / FTFSGTG
            predictions = self.src_user_gp.dot(self.pattern).dot(self.src_item_gp.T)

            loss = sqrt(np.sum((self.src_mat - predictions)**2) /
                   np.count_nonzero(self.src_mat))
            print("Reconstruction error in iteration {0} is {1:.4f}".format(run, loss))
        # Membership binary
        u_aux = self.binary_transform(self.src_user_gp)
        v_aux = self.binary_transform(self.src_item_gp)
        # Construct code book
        nominator = (u_aux.T.dot(self.src_mat)).dot(v_aux)
        denumerator = u_aux.T.dot(np.ones((self.src_mat.shape[0], 1))).dot(np.ones((
                self.src_mat.shape[1], 1)).T.dot(v_aux))
        self.pattern = np.divide(nominator, denumerator)
        print("rating pattern: \n{}".format(self.pattern))

    @staticmethod
    def binary_transform(membership_mat):
        """
        Change membership matrix to binary matrix.
        Especially for each row, find largest value and replace it to 1, while change others to 0.

        Parameters
        -----------------
        membership_mat : numpy.ndarray
            A 2 * D membership matrix (user-/item-group membership matrix)

        Returns
        -----------------
        A binaryzation matrix.
        """
        aux = np.zeros(membership_mat.shape)
        # For each row, find the column which has largest value
        largest_cols = np.argmax(membership_mat, axis=1)
        assert (largest_cols.shape[0] == membership_mat.shape[0])
        for row in xrange(membership_mat.shape[0]):
            # Replace largest value in this row with 1, while keep others to be 0
            aux[row, largest_cols[row]] = 1.0
        return aux

    def cb_transfer(self):
        """
        Transfer constructed Code Book to target domain
        to fill missing values in the training data.
        """
        for row in xrange(self.tar_train_mat.shape[0]):
            # randomly select one column
            col = randint(0, self.item_cluster - 1)
            self.tar_user_gp[row, col] = 1.

        for run in xrange(100):
            for i in xrange(self.tar_train_mat.shape[0]):
                tmp = self.tar_train_mat[i, :] - self.indicator[i, :].multiply(self.pattern.dot(
                        self.tar_item_gp.T))
                tmp = tmp.dot(tmp.T)
                # Find the column that has smallest value
                j_ = np.argmin(np.diagonal(tmp))
                col_selector = [x for x in xrange(self.user_cluster) if x != j_]
                self.tar_user_gp[i, j_] = 1.0
                self.tar_user_gp[i, col_selector] = 0.0

            for i in xrange(self.tar_train_mat.shape[1]):
                tmp = self.tar_train_mat.tocsc()[:, i] - \
                      self.indicator.tocsc()[:, i].multiply(self.tar_user_gp.dot(self.pattern))
                tmp = tmp.T.dot(tmp)
                # Find the column that has smallest value
                j_ = np.argmin(np.diagonal(tmp))
                col_selector = [x for x in xrange(self.item_cluster) if x != j_]
                self.tar_item_gp[i, j_] = 1.
                self.tar_item_gp[i, col_selector] = 0.
        # Fill in missing values
        self.tar_train_mat = self.indicator.multiply(self.tar_train_mat) + \
                             np.multiply((np.ones(self.indicator.shape) - self.indicator),
                             self.tar_user_gp.dot(self.pattern).dot(self.tar_item_gp.T))

    def knn_predict(self):
        """
        Call KNN user/item-based collaborative filtering to make predictions.
        """
        # Compute similarity matrix
        sim = self.pcc_similarity(self.tar_train_mat)
        # Select top k users/items to make predictions.
        pred = self.predict_topk(self.tar_train_mat, sim, kind='user', k=self.n_neighbor)
        # Compute mae and rmse on test set
        mae = np.sum(np.abs((self.tar_test_mat.data - pred[
            self.tar_test_mat.nonzero()]))) / self.tar_test_mat.nnz
        rmse = sqrt(np.sum(np.power((self.tar_test_mat.data - pred[
            self.tar_test_mat.nonzero()]), 2)) / self.tar_test_mat.nnz)
        return mae, rmse

    @staticmethod
    def cosine_similarity(ratings, kind='user', epsilon=1e-9):
        """
        Compute user-user/item-item similarity matrix by cosine measure.

        Parameters
        -----------------
        ratings : n_samples * m_features sparse csr_matrix
        The input rating matrix.

        kind : str, default 'user'
        The indicator to denote making user-user or item-item similarity matrix.

        epsilon : float, default 1e-9
        small number for handling dived-by-zero errors

        Returns
        -----------------
        A n * n user-user/item-item similarity matrix.
        """
        if kind == 'user':
            sim = ratings.dot(ratings.T) + epsilon
        elif kind == 'item':
            sim = ratings.T.dot(ratings) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return sim / norms / norms.T

    @staticmethod
    def pcc_similarity(ratings, kind='user'):
        """
        Compute user-user/item-item similarity matrix by Pearson Correlation Coefficient.

        Parameters
        ---------------
        ratings : n_samples * m_features sparse csr_matrix
        The input rating matrix.

        kind : str, default 'user'
        The indicator to denote making user-user or item-item similarity matrix.

        Returns
        ---------------
        A n * n ndarray user-user/item-item similarity matrix.
        """
        if kind == 'user':
            sim = CBT.corrcoef_csr(ratings)
        elif kind == 'item':
            sim = CBT.corrcoef_csr(ratings, axis=0)
        return sim

    @staticmethod
    def corrcoef_csr(x, axis=1):
        """
        Compute correlation matrix for sparse matrix.

        Parameters
        ----------------------
        x: Sparse csr_matrix
        A n * m sparse matrix, where row is sample and column is feature.

        axis : int, default 1.
        The indicator to denote generating a n * n or m * m correlation matrix.
        if axis=1, generate a n * n correlation matrix;
        if axis=0, generate a m * m correlation matrix.

        Returns
        ---------------------
        A n * n or m * m ndarray correlation matrix.
        """
        covx = CBT.cov_csr(x, axis=axis)
        stdx = np.sqrt(np.diag(covx))[np.newaxis, :]
        return covx / (stdx.T * stdx)

    @staticmethod
    def cov_csr(x, axis=1):
        """
        Compute covariance matrix for a sparse matrix, assumes row represents sample.

        Parameters
        ----------------------
        x: scipy.sparse.csr_matrix
        A n * m sparse matrix, where row is sample and column is feature.

        axis : int, default 1.
        The indicator to denote generating a n * n or m * m correlation matrix.

        Returns
        ---------------------
        A n * n or m * m ndarray covariance matrix.
        """
        meanx = x.sum(axis=axis) / float(x.shape[axis])
        if axis == 0:
            return np.array((x.T * x) / x.shape[axis] - meanx.T * meanx)
        else:
            return np.array((x * x.T) / x.shape[axis] - meanx * meanx.T)

    @staticmethod
    def predict_topk(ratings, similarity, kind='user', k=20):
        """
        K Nearest Neighbor based Collaborative filtering.

        Parameters:
        -----------------------------
        ratings : numpy.ndarray
        A n * m dense rating matrix.

        similarity : numpy.ndarray
        A user-user/item-item similarity matrix.

        kind : str, default 'user'
        Indicator for choosing user-based or item-based CF.

        k : int, default 20
        The number of neighbors.

        Returns
        ------------------------
        A n * m numpy.ndarray predicted rating matrix.
        """
        # ratings = ratings.toarray()
        pred = np.zeros(ratings.shape)
        if kind == 'user':
            for i in xrange(ratings.shape[0]):
                top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
                for j in xrange(ratings.shape[1]):
                    pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                    pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        if kind == 'item':
            for j in xrange(ratings.shape[1]):
                top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
                for i in xrange(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

        return pred
