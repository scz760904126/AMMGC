from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp

import networkx as nx
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random


#
# flags = tf.app.flags
# FLAGS = flags.FLAGS

def construct_self_feed_dict(emb, train_drug_miRNA_matrix, positive_mask, negative_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['emb']: emb})
    feed_dict.update({placeholders['adj_label']: train_drug_miRNA_matrix})
    feed_dict.update({placeholders['positive_mask']: positive_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict


def construct_attention_feed_dict(emb, train_drug_miRNA_matrix, positive_mask, negative_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['emb'][i]: emb[i] for i in range(len(emb))})
    feed_dict.update({placeholders['adj_label']: train_drug_miRNA_matrix})
    feed_dict.update({placeholders['positive_mask']: positive_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict


def constructNet(miRNA_dis_matrix):
    miRNA_matrix = np.mat(np.zeros((miRNA_dis_matrix.shape[0], miRNA_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.mat(np.zeros((miRNA_dis_matrix.shape[1], miRNA_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((miRNA_matrix, miRNA_dis_matrix))
    mat2 = np.hstack((miRNA_dis_matrix.T, dis_matrix))

    return np.vstack((mat1, mat2))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm


def matrix_normalize(similarity_matrix):
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        for i in range(similarity_matrix.shape[0]):
            similarity_matrix[i, i] = 0
        for i in range(200):
            D = np.diag(np.array(np.sum(similarity_matrix, axis=1)).flatten())  # 求得每一行的sum，再使其对角化
            D = np.linalg.pinv(np.sqrt(D))  # 开方，再取伪逆矩阵
            similarity_matrix = D * similarity_matrix * D
    else:
        for i in range(similarity_matrix.shape[0]):
            if np.sum(similarity_matrix[i], axis=1) == 0:
                similarity_matrix[i] = similarity_matrix[i]
            else:
                similarity_matrix[i] = similarity_matrix[i] / np.sum(similarity_matrix[i], axis=1)
    return similarity_matrix


def masked_bilinearsigmoid_cross_entropy(preds, labels, mask, negative_mask):
    """Softmax cross-entropy loss with masking."""

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    mask = tf.reshape(mask, shape=[79924])
    loss *= mask
    return tf.reduce_mean(loss)


def gcn_masked_softmax_cross_entropy(preds, labels, positive_mask, negative_mask, pos_weight):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=preds, pos_weight=pos_weight)

    # preds = tf.cast(preds, tf.float32)
    # labels = tf.cast(labels, tf.float32)
    # loss = tf.square(preds - labels)

    positive_mask += negative_mask
    # print(mask)
    mask = tf.cast(positive_mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    mask = tf.reshape(mask, shape=[79924])

    loss *= mask
    return tf.reduce_mean(loss)


def generate_mask(train_drug_miRNA_matrix, N):
    num = 0
    mask = np.zeros(train_drug_miRNA_matrix.shape)
    while (num < 1 * N):
        a = random.randint(0, 105)
        b = random.randint(0, 753)
        if train_drug_miRNA_matrix[a, b] != 1 and mask[a, b] != 1:
            mask[a, b] = 1
            num += 1
    mask = np.reshape(mask, [-1, 1])
    return mask


def load_data(train_arr, test_arr):
    """Load data."""
    labels = np.loadtxt("drug-miRNA.txt")

    logits_test = sp.csr_matrix((labels[test_arr, 2], (labels[test_arr, 0] - 1, labels[test_arr, 1] - 1)),
                                shape=(106, 754)).toarray()
    logits_test = logits_test.reshape([-1, 1])
    #     logits_test = np.hstack((logits_test,1-logits_test))

    logits_train = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0] - 1, labels[train_arr, 1] - 1)),
                                 shape=(106, 754)).toarray()
    logits_train = logits_train.reshape([-1, 1])
    # logits_temp_train = logits_train
    #
    # train_list = []
    # train_list.append(logits_temp_train)

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])
    # train_mask = np.array(logits_train[:, 0]).reshape([-1, 1])
    # test_mask = np.array(logits_test[:, 0]).reshape([-1, 1])

    M = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0] - 1, labels[train_arr, 1] - 1)),
                      shape=(106, 754)).toarray()

    return logits_train, logits_test, train_mask, test_mask, labels


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_norm, adj_label, features, positive_mask, negative_mask, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj_norm']: adj_norm})
    feed_dict.update({placeholders['adj_label']: adj_label})
    feed_dict.update({placeholders['positive_mask']: positive_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict


def masked_cross_entropy(preds, labels, label_mask, test_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)

    error = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds)
    # pos_weight = 1
    # error = tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=pos_weight)
    label_mask += test_mask
    mask = tf.cast(label_mask, dtype=tf.float32)
    mask = tf.reshape(mask, shape=[79924])

    error *= mask

    return tf.sqrt(tf.reduce_mean(error))


def masked_accuracy(preds, labels, label_mask, test_mask):
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds - labels)
    label_mask += test_mask
    mask = tf.cast(test_mask, dtype=tf.float32)
    mask = tf.reshape(mask, shape=[79924])
    error *= mask

    return tf.sqrt(tf.reduce_mean(error))
