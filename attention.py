from layers import *
from metrics import *
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS


def masked_cross_entropy(preds, labels, label_mask, test_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)

    # error = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    pos_weight = 1
    error = tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=pos_weight)
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
        self.att_ls = []

        self.inputs = None
        self.att = None
        self.feedforward = None
        self.mixed = None
        # self.output = None

        self.pred = 0
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
            hidden, att_ = layer(self.activations[-1])
            self.activations.append(hidden)
            self.att_ls.append(att_)

        self.output = self.activations[-1]
        self.att = self.att_ls[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics

        self.predict()
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        raise NotImplementedError

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


class attentionModel(Model, ):
    def __init__(self, placeholders, num_edges, **kwargs):
        super(attentionModel, self).__init__(**kwargs)

        self.num_edges = num_edges
        self.output_dim = 860
        self.inputs = placeholders['emb']
        self.adjs = placeholders['emb']
        self.labels = placeholders['adj_label']
        self.positive_mask = placeholders['positive_mask']
        self.negative_mask = placeholders['negative_mask']
        self.num_support = len(self.adjs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.output = None
        self.build()
        self.predict()

    def _build(self):
        self.layers.append(attention(outputdim=self.output_dim,
                                     adjs=self.adjs,
                                     placeholders=self.placeholders,
                                     ))

    def predict(self):
        num_drug = 106
        pred = tf.matmul(self.output, self.output, transpose_b=True)
        self.pred = tf.slice(pred, [0, num_drug], [num_drug, -1])
        self.pred = tf.reshape(self.pred, [-1])
        return self.pred

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        pos_weight = float(79924 - self.num_edges) / self.num_edges

        self.loss += gcn_masked_softmax_cross_entropy(self.pred, tf.reshape(self.labels, [-1]), self.positive_mask,
                                                      self.negative_mask, pos_weight=pos_weight)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.pred, tf.reshape(self.labels, [-1]), self.positive_mask,
                                        self.negative_mask)
