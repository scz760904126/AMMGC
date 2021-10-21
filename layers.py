from inits import *
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Self_attention(Layer):
    """Self_attention layer."""
    def __init__(self,size,n_head,size_per_head,placeholders,act=tf.nn.relu, **kwargs):
        super(Self_attention, self).__init__(**kwargs)
        self.n_head = n_head
        self.size_per_head = size_per_head
        self.placeholders = placeholders
        self.act = act

        with tf.variable_scope(self.name + '_vars'):
            self.vars['Q_weight'] = glorot([size, size_per_head])
            self.vars['K_weight'] = glorot([size, size_per_head])
            self.vars['V_weight'] = glorot([size, size_per_head])
            self.vars['O_weight'] = glorot([size, n_head * size_per_head])
        if self.logging:
            self._log_vars()

    def _call(self,inputs):
        att_list = []
        Q = inputs[0]
        K = inputs[1]
        V = inputs[2]

        Q = tf.cast(Q, dtype=tf.float32)
        K = tf.cast(K, dtype=tf.float32)
        V = tf.cast(V, dtype=tf.float32)

        for i in range(self.n_head):
            Q_temp = tf.matmul(Q, self.vars['Q_weight'])
            K_temp = tf.matmul(K, self.vars['K_weight'])
            V_temp = tf.matmul(V, self.vars['V_weight'])
            A_temp = tf.matmul(Q_temp, K_temp,transpose_b=True) \
                     / tf.sqrt(float(self.size_per_head))
            A_temp = tf.nn.softmax(A_temp)
            att = tf.matmul(A_temp, V_temp)
            att_list.append(att)

        O = tf.concat(att_list, axis=1, name='concat')
        O = tf.cast(O, tf.float32)
        O = tf.matmul(O, self.vars['O_weight'])

        return O


class feedforward(Layer):
    """feedforward layer."""
    def __init__(self,placeholders,num_node,size,act=tf.nn.relu, **kwargs):
        super(feedforward, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.act = act

        with tf.variable_scope(self.name + '_vars'):
            self.vars['W1'] = glorot([size, size])
            self.vars['W2'] = glorot([size, size])
            self.vars['b1'] = glorot([num_node, size])
            self.vars['b2'] = glorot([num_node, size])

        if self.logging:
            self._log_vars()

    def _call(self,att):

        forward1 = tf.matmul(att, self.vars['W1'])
        forward1 = tf.add(forward1,self.vars['b1'])
        forward1 = self.act(forward1)

        forward2 = tf.matmul(forward1, self.vars['W2'])
        forward2 = tf.add(forward2, self.vars['b1'])

        return forward2

class attention(Layer):
    """attention layer."""
    def __init__(self,outputdim,adjs,placeholders,act=tf.nn.relu, **kwargs):
        super(attention, self).__init__(**kwargs)

        self.output_dim = outputdim
        self.adjs = adjs
        self.placeholders = placeholders
        self.num_support = len(self.adjs)
        self.act = act

        with tf.variable_scope(self.name + '_vars'):
            self.vars['attWeight'] = glorot([self.num_support, self.output_dim])
            self.vars['attBias'] = glorot([self.num_support, 128])

        if self.logging:
            self._log_vars()

    def _call(self,inputs):

        attention = []
        attADJ = []
        for i in range(self.num_support):
            tmpattention = tf.matmul(tf.reshape(self.vars['attWeight'][i], [1, -1]), self.adjs[i]) \
                           + tf.reshape(self.vars['attBias'][i], [1, -1])
            attention.append(tmpattention)

        attentions = tf.concat(attention, 0)
        attention = tf.nn.softmax(attentions, 0)
        for i in range(self.num_support):
            attADJ.append(tf.matmul(self.adjs[i], tf.diag(attention[i])))

        # mixedADJ = tf.add_n(attADJ)
        mixedADJ = tf.concat(attADJ, 1)

        return mixedADJ



