#!user/bin/ env python
# Author: Peng Hao
# Email: peng.hao@student.uts.edu.au

import pandas as pd
import scipy.sparse as ssp
import numpy as np
import os
import random
import scipy.io as sio
PATH = os.path.dirname(os.path.realpath(__file__)) + '/../source/'


def fill_row_miss_value(arr_csr):
    """
    For each row in arr_csr,
    fill the missing value with mean value of non-zero elements in this row.

    Parameters
    ----------------
    arr_csr : sparse.csr_matrix
    The input sparse matrix.

    Returns
    ----------------
    Dense 2*D array after filling values to missing positions.
    """
    arr = arr_csr.toarray()
    all_cols = np.arange(arr_csr.shape[1])
    for i in xrange(len(arr_csr.indptr) - 1):
        row_sum = arr_csr[i, :].sum() / float(arr_csr[i, :].nnz)
        start = arr_csr.indptr[i]
        end = arr_csr.indptr[i + 1]
        arr[i, np.setdiff1d(all_cols, arr_csr.indices[start: end])] = row_sum
    return arr


def parse():
    # Load source data: EachMovie
    src_original = sio.loadmat(PATH + 'eachmovie.mat')['eachmovie']
    src_original = ssp.csr_matrix(src_original)
    # Randomly select 500 users and items
    src_part_users = random.sample(range(src_original.shape[0]), 500)
    src_part_items = random.sample(range(src_original.shape[1]), 500)
    # Slicing source matrix
    src_original = src_original[src_part_users, :].tocsc()[:, src_part_items].tocsr()
    src_original.eliminate_zeros()
    src_user_num = src_original.shape[0]
    src_item_num = src_original.shape[1]
    # Count statistics of source dataset
    print("Source rating matrix shape: {0} * {1}".format(src_user_num, src_item_num))
    print("Sparsity in source dataset: {:.4f}".format(100. * float(src_original.nnz) / (
        src_user_num * src_item_num)))
    print(20 * '--')
    # Fill missing values
    src_mat = fill_row_miss_value(src_original)

    # Load target data: ML-100K
    tar_data_path = PATH + 'u.data'
    names = ['user_id', 'item_id', 'rating', 'ts']
    data = pd.read_csv(tar_data_path, names=names, sep='\t')
    del data['ts']
    # Remove users who rated less than 40 movies
    group_object = data.groupby('user_id')
    data = data[group_object.item_id.transform(len) > 40]
    # Remove movies that were rated by less than 10 users
    group_object = data.groupby('item_id')
    data = data[group_object.user_id.transform(len) > 10]

    users = list(data.user_id.unique())
    items = list(data.item_id.unique())
    # Randomly select 500 users and 1000 items
    select_users = random.sample(users, 500)
    select_items = random.sample(items, 1000)
    # Prune data
    data = data[data.user_id.isin(select_users)]
    data = data[data.item_id.isin(select_items)]
    # Make rating matrix
    tar_mat = pd.pivot_table(data, index=['user_id'], columns=['item_id'], values='rating')
    # Fill NaN with 0
    tar_mat = tar_mat.fillna(0)
    # Change into sparse structure
    tar_mat = ssp.csr_matrix(tar_mat.values)
    # Count statistics of target dataset
    tar_user_num = tar_mat.shape[0]
    tar_item_num = tar_mat.shape[1]
    print ("Target rating matrix shape: {0} * {1}".format(tar_user_num, tar_item_num))
    sparsity = float(tar_mat.nnz)
    sparsity /= (tar_user_num * tar_item_num)
    print("Sparsity in target dataset: {:.4f}%".format(sparsity * 100.))
    print(20 * '--')

    return src_mat, tar_mat


if __name__ == '__main__':
    parse()
