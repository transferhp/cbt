#!user/bin/ env python
# Author: Peng Hao
# Email: peng.hao@student.uts.edu.au

import numpy as np
from data.parse_data import parse
from model.cbt import CBT


def run():
    # Keep result of each run
    mae_result = []
    rmse_result = []
    for r in xrange(10):
        # Load data
        src_mat, tar_mat = parse()
        # Splitting train and test set for target data
        tar_train, tar_test = train_test_split(tar_mat, k=300, given_k=10)
        # Define user group number
        k = 50
        # Define item group number
        l = 50
        # Call CBT model to run
        cbt = CBT(tar_train, tar_test, src_mat, k, l)
        mae, rmse = cbt.fit()
        print("mae in iteration-{0} is: {1:.4f}".format(r + 1, mae))
        print("rmse in iteration-{0} is: {1:.4f}".format(r + 1, rmse))
        mae_result.append(mae)
        rmse_result.append(rmse)
    print("Avg mae = {:.4f}".format(np.mean(mae_result)))
    print("Avg rmse = {:.4f}".format(np.mean(rmse_result)))


def train_test_split(ratings, k, given_k):
    """
    Split input data into training and test data.
    The first k row users and their ratings are selected as training data,
    the remaining users and their ratings are left to test data.
    For each test user, randomly select given_k observed ratings and provide them to training set,
    and the remaining ratings are used for evaluation.


    Parameters
    ----------------
    ratings : sparse csr_matrix
    The input n * m rating matrix.

    k : int
    Top k users are selected for training set,
    then n - k users are left for test.

    given_k : int
    For each test user, give_k observed ratings are provided to training.

    Returns
    -----------------
    train : sparse csr_matrix
    A n * m rating matrix for training data.

    test : sparse csr_matrix
    A n * m rating matrix for test data.

    """
    train = ratings.copy()
    train.data[train.indptr[k]:] = 0.  # only keep first k rows
    train.eliminate_zeros()

    test = ratings.copy()
    test.data[test.indptr[0]: test.indptr[k]] = 0.  # only keep remaining rows
    test.eliminate_zeros()

    # change sparsity structure
    train = train.tolil()
    test = test.tolil()

    for row in xrange(k, test.shape[0]):
        col_indices = np.random.choice(test[row, :].nonzero()[1],
                                       size=given_k,
                                       replace=False)
        for col in col_indices:
            train[row, col] = test[row, col]
            test[row, col] = 0.

    train = train.tocsr()
    test = test.tocsr()

    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)  # sparse format
    return train, test


if __name__ == '__main__':
    run()
