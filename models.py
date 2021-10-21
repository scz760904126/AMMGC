from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp
from utils import *
import networkx as nx
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.hid = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # activations

        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.outputs = self.activations[-1]
        self.hid = self.activations[-2]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def hidd(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            # inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            # x = tf.transpose(inputs)
            # x = tf.matmul(inputs, x)
            # x = tf.reshape(x, [-1])
            # outputs = self.act(x)

            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            num_drug = 106
            # drug_matrix = inputs[0:num_drug,:]
            # miRNA_matrix = inputs[num_drug::,:]
            drug_matrix = tf.slice(inputs, [0, 0], [num_drug, -1])
            miRNA_matrix = tf.slice(inputs, [num_drug, 0], [860 - num_drug, -1])
            # x = tf.transpose(inputs)
            miRNA_matrix = tf.transpose(miRNA_matrix)
            x = tf.matmul(drug_matrix, miRNA_matrix)
            x = tf.reshape(x, [-1])
            # outputs = self.act(x)
        return x


class GCNModel():
    def __init__(self, placeholders, num_features, features_nonzero, num_nodes, num_edges,name):
        self.name = name
        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj_norm']
        self.dropout = placeholders['dropout']

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        # self.mask = placeholders['labels_mask']
        # self.negative_mask = placeholders['negative_mask']


        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout)(self.inputs)

        self.embeddings = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout)(self.hidden1)

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=FLAGS.hidden2,
            act=lambda x: x)(self.embeddings)

        pos_weight = float(self.num_nodes ** 2 - self.num_edges) / self.num_edges
        norm = self.num_nodes ** 2 / float((self.num_nodes ** 2 - self.num_edges) * 2)
        # pos_weight = float(79924 - self.num_edges) / self.num_edges
        # norm = 79924 / float((79924 - self.num_edges) * 2)

        pos_weight = 1
        self.cost = gcn_masked_softmax_cross_entropy(self.reconstructions, tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_label'], validate_indices=False),
                                   [-1]), self.placeholders['positive_mask'],self.placeholders['negative_mask'],  pos_weight)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

