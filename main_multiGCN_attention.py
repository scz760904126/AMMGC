from __future__ import division
from __future__ import print_function
from datetime import datetime
from models import *
from utils import *
from metrics import *
from attention import attentionModel
import tensorflow.compat.v1 as tf
import random
import matplotlib.pyplot as plt


tf.disable_eager_execution()

def GCN_process(train_drug_miRNA_matrix, adj_train, adj_norm, features, num_nodes, num_edges, positive_train,
                positive_mask):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.') # 100 epochs
    flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')

    num_drug = 106
    num_miRNA = 754
    features = normalize_features(features)
    features = sparse_to_tuple(sp.coo_matrix(features))

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj_norm': tf.sparse_placeholder(tf.float32),
        'adj_label': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'positive_mask': tf.placeholder(shape=[79924, 1], dtype=tf.int32),
        'negative_mask': tf.placeholder(shape=[79924, 1], dtype=tf.int32),
    }

    # Create model
    model = GCNModel(placeholders, num_features, features_nonzero, num_nodes, num_edges, name='yeast_gcn')

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = adj_label.todense()
    adj_label = adj_label[0:106, 106::]
    adj_label = sp.csr_matrix(adj_label)
    adj_label = sparse_to_tuple(adj_label)

    epoches = []
    avg_costs = []

    # Train model
    for epoch in range(FLAGS.epochs):
        # Create optimizer
        negative_mask = generate_mask(train_drug_miRNA_matrix, len(positive_train))

        t = time.time()

        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, positive_mask, negative_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # One update of parameter matrices

        _, avg_cost = sess.run([model.opt_op, model.cost], feed_dict=feed_dict)

        epoches.append(epoch)
        avg_costs.append(avg_cost)
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(avg_cost),
              "time=", "{:.5f}".format(time.time() - t))

    print('Optimization Finished!')

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.embeddings, feed_dict=feed_dict)

    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)

        # delete all of flags before running the main command

    del_all_flags(flags.FLAGS)

    return emb


def attention_process(emb, positive_train, positive_mask, train_drug_miRNA_matrix, num_edges):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('model', 'attSemiGAE', 'Model string.')  # 'gcn', 'semiencoder', 'attSemiGAE'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')    # 300 epochs
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    num_supports = len(emb)
    # Load data
    placeholders = {
        'emb': [tf.placeholder(tf.float32, shape=(860, 128)) for _ in range(num_supports)],
        'adj_label': tf.placeholder(tf.float32, shape=(106, 754)),
        'positive_mask': tf.placeholder(shape=[79924, 1], dtype=tf.int32),
        'negative_mask': tf.placeholder(shape=[79924, 1], dtype=tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }

    # Create model
    model = attentionModel(placeholders, num_edges, logging=True)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    epoches = []
    avg_costs = []

    # Train model
    for epoch in range(FLAGS.epochs):
        negative_mask = generate_mask(train_drug_miRNA_matrix, len(positive_train))
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_attention_feed_dict(emb, train_drug_miRNA_matrix, positive_mask, negative_mask,
                                                  placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

        epoches.append(epoch)
        avg_costs.append(outs[1])

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "time=",
              "{:.5f}".format(time.time() - t))

    # name = 'gcn_result_con/5_fold_epoches.csv'
    # np.savetxt(name, epoches, delimiter=',')
    # name = 'gcn_result_con/5_fold_avg_costs.csv'
    # np.savetxt(name, avg_costs, delimiter=',')

    print("Optimization Finished!")

    feed_dict.update({placeholders['dropout']: 0})
    output = sess.run(model.output, feed_dict=feed_dict)

    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    del_all_flags(flags.FLAGS)

    return output


# 交叉验证
def cross_validation(adj, seed):
    num_nodes = adj.shape[0]

    drug_feature_name1 = 'features/drug_feature_matrix.txt'
    drug_feature1 = np.loadtxt(drug_feature_name1, dtype=float)
    drug_feature_name2 = 'features/drug_labelencoding.txt'
    drug_feature2 = np.loadtxt(drug_feature_name2, dtype=float)
    miRNA_feature_name1 = 'features/ncrna_expression_full.txt'
    miRNA_feature1 = np.loadtxt(miRNA_feature_name1, dtype=float)
    miRNA_feature_name2 = 'features/ncrna_GOsimilarity_full.txt'
    miRNA_feature2 = np.loadtxt(miRNA_feature_name2, dtype=float)

    features1 = np.vstack((drug_feature1, np.hstack((np.zeros(
        shape=(miRNA_feature1.shape[0], drug_feature1.shape[1] - miRNA_feature1.shape[1]), dtype=int),
                                                     miRNA_feature1))))
    features2 = np.vstack((np.hstack((np.zeros(
        shape=(drug_feature1.shape[0], miRNA_feature2.shape[1] - drug_feature1.shape[1]), dtype=int), drug_feature1)),
                           miRNA_feature2))

    features3 = np.vstack((np.hstack((np.zeros(
        shape=(drug_feature2.shape[0], miRNA_feature1.shape[1] - drug_feature2.shape[1]), dtype=int), drug_feature2)),
                           miRNA_feature1))
    features4 = np.vstack((np.hstack((np.zeros(
        shape=(drug_feature2.shape[0], miRNA_feature2.shape[1] - drug_feature2.shape[1]), dtype=int), drug_feature2)),
                           miRNA_feature2))

    num_drug = 106
    drug_miRNA_matrix = adj.todense()[0:num_drug, num_drug::]
    none_zero_position = np.where(drug_miRNA_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    np.random.seed(seed)
    positive_randomlist = [i for i in range(len(none_zero_row_index))]
    random.shuffle(positive_randomlist)

    sum_metric = np.zeros((1, 7))
    k_folds = 5
    print("seed=%d, evaluating miRNA-disease...." % (seed))

    for k in range(k_folds):
        metric = np.zeros((1, 7))
        print("------this is %dth cross validation------" % (k + 1))
        if k != k_folds - 1:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds):(k + 1) * int(
                len(none_zero_row_index) / k_folds)]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))

        else:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds)::]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))

        positive_test_row = none_zero_row_index[positive_test]
        positive_test_col = none_zero_col_index[positive_test]

        train_drug_miRNA_matrix = np.copy(drug_miRNA_matrix)
        train_drug_miRNA_matrix[positive_test_row, positive_test_col] = 0
        positive_mask = train_drug_miRNA_matrix.reshape(-1, 1)

        num_edges = train_drug_miRNA_matrix.sum()
        print('训练集中边的数目')
        print(num_edges)

        adj_train = constructNet(train_drug_miRNA_matrix)
        adj_train = sp.csr_matrix(adj_train)
        # 得到训练矩阵
        adj = adj_train
        adj_norm = preprocess_graph(adj)

        tf.reset_default_graph()

        emb1 = GCN_process(train_drug_miRNA_matrix, adj_train, adj_norm, features1, num_nodes, num_edges,
                           positive_train, positive_mask)
        emb2 = GCN_process(train_drug_miRNA_matrix, adj_train, adj_norm, features2, num_nodes, num_edges,
                           positive_train, positive_mask)
        emb3 = GCN_process(train_drug_miRNA_matrix, adj_train, adj_norm, features3, num_nodes, num_edges,
                           positive_train, positive_mask)
        emb4 = GCN_process(train_drug_miRNA_matrix, adj_train, adj_norm, features4, num_nodes, num_edges,
                           positive_train, positive_mask)

        emb = []
        emb.append(emb1)
        emb.append(emb2)
        emb.append(emb3)
        emb.append(emb4)

        tf.reset_default_graph()
        att_emb = attention_process(emb, positive_train, positive_mask, train_drug_miRNA_matrix, num_edges)

        name = 'gcn_result_con/5_fold_att_emb.csv'
        np.savetxt(name, att_emb, delimiter=',')

        adj_rec = np.dot(att_emb, att_emb.T)
        adj_rec = adj_rec[0: num_drug, num_drug::]

        positive_test_row = none_zero_row_index[positive_test]
        positive_test_col = none_zero_col_index[positive_test]
        negative_position = np.where(drug_miRNA_matrix == 0)
        negative_test_row = negative_position[0]
        negative_test_col = negative_position[1]

        test_row = np.append(positive_test_row, negative_test_row)
        test_col = np.append(positive_test_col, negative_test_col)
        test_real = []
        test_pre = []
        for i in range(len(test_row)):
            label = drug_miRNA_matrix[test_row[i], test_col[i]]
            score = sigmoid(adj_rec[test_row[i], test_col[i]])
            test_real.append(label)
            test_pre.append(score)
        test_real = np.array(test_real)
        test_pre = np.array(test_pre)

        metric = model_evaluate(test_real, test_pre)
        print("------the metrics of %dth cross validation------" % (k + 1))
        print(metric)

        sum_metric += metric
    return sum_metric / k_folds


datetime1 = datetime.now()
name = 'drug-miRNA.txt'
bi_adj = np.zeros((106, 754))
index = np.loadtxt(name, dtype=int)
row = index[:, 0] - 1
col = index[:, 1] - 1
bi_adj[row, col] = 1
adj_dense = constructNet(bi_adj)
adj = sp.csr_matrix(adj_dense)

sum_metric = np.zeros((1, 7))
circle = 10
for i in range(circle):
    metric = np.zeros((1, 7))
    metric = cross_validation(adj, i)
    sum_metric += metric
    name = 'gcn_result_con/128_128_multi_attention_5fold_drug_miRNA_seed' + str(i) + '.csv'
    np.savetxt(name, metric, delimiter=',')
avg_metric = sum_metric / circle
name = 'gcn_result_con/avg_128_128_multi_attention_5fold_drug_miRNA.csv'
np.savetxt(name, avg_metric, delimiter=',')
print('##########running time###############')
print(datetime.now() - datetime1)
