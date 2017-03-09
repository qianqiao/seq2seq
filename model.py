import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn
from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from output_projection import output_projection_layer

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Seq2SeqModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            vocab=None,
            embed=None,
            learning_rate=0.5,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=False):

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.loss = {}

        # build the vocab table (symbol to index)
        if vocab != None:
            self.symbols = tf.Variable(vocab, trainable=False, name="symbols")
        else:
            self.symbols = tf.Variable(np.array(['.']*num_symbols), name="symbols")
        self.symbol2index = HashTable(KeyValueTensorInitializer(self.symbols,
            tf.Variable(np.array([i for i in range(num_symbols)], dtype=np.int32), False)),
            default_value=UNK_ID, name="symbol2index")

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        # define the inputs of model
        self.inputs = {}
        self.inputs['post'] = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.inputs['post_len'] = tf.placeholder(tf.int32, shape=(None))  # batch
        self.inputs['resp'] = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.inputs['resp_len'] = tf.placeholder(tf.int32, shape=(None))  # batch
        #self.inputs['part_post'] = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        #self.inputs['part_post_len'] = tf.placeholder(tf.int32, shape=(None))  # batch
        self.inputs['hist_post'] = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.inputs['hist_post_len'] = tf.placeholder(tf.int32, shape=(None))  # batch
        self.inputs['hist_resp'] = tf.placeholder(tf.string, shape=(None, None))  # batch*len
        self.inputs['hist_resp_len'] = tf.placeholder(tf.int32, shape=(None))  # batch
        self.inputs['gate'] = tf.placeholder(tf.float32, shape=[None])   # batch
        gate = tf.reshape(self.inputs['gate'], [-1, 1])

        # symbol to index
        post = self.symbol2index.lookup(self.inputs['post'])   # batch*len
        resp = self.symbol2index.lookup(self.inputs['resp'])   # batch*len
        #part_post = self.symbol2index.lookup(self.inputs['part_post'])   # batch*len
        hist_post = self.symbol2index.lookup(self.inputs['hist_post'])   # batch*len
        hist_resp = self.symbol2index.lookup(self.inputs['hist_resp'])   # batch*len

        batch_size, resp_max_len = tf.shape(resp)[0], tf.shape(resp)[1]

        # calculate the input and mask for decoder
        resp_shift = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
            tf.split(resp, [resp_max_len-1, 1], 1)[0]], 1)   # batch*len
        resp_mask = tf.reshape(tf.cumsum(tf.one_hot(self.inputs['resp_len']-1,
            resp_max_len), reverse=True, axis=1), [-1, resp_max_len])   # batch*len

        # index to embedding
        encoder_input = tf.nn.embedding_lookup(self.embed, post)  # batch*len*unit
        decoder_input = tf.nn.embedding_lookup(self.embed, resp_shift)   # batch*len*unit
        hist_post_embed = tf.nn.embedding_lookup(self.embed, hist_post)   # batch*len*unit
        hist_resp_embed = tf.nn.embedding_lookup(self.embed, hist_resp)   # batch*len*unit

        # choose GRU or LSTM and the number of layers
        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units)] * num_layers)
        else:
            cell = MultiRNNCell([GRUCell(num_units)] * num_layers)

        # rnn encoder
        with tf.variable_scope('encoder'):
            encoder_output, encoder_state = dynamic_rnn(cell, encoder_input,
                    self.inputs['post_len'], dtype=tf.float32)
        with tf.variable_scope('encoder', reuse=True):
            hist_post_output, _ = dynamic_rnn(cell, hist_post_embed,
                    self.inputs['hist_post_len'], dtype=tf.float32)
        with tf.variable_scope('encoder', reuse=True):
            hist_resp_output, _ = dynamic_rnn(cell, hist_resp_embed,
                    self.inputs['hist_resp_len'], dtype=tf.float32)
        hist_output = tf.concat([hist_post_output, hist_resp_output], 1)

        # get attention function
        with tf.variable_scope('attn_encoder'):
            attn_encoder_keys, attn_encoder_values, attn_encoder_score_fn,\
                    attn_encoder_construct_fn = attention_decoder_fn.prepare_attention(
                            encoder_output, 'luong', num_units)
        with tf.variable_scope('attn_context'):
            attn_hist_keys, attn_hist_values, attn_hist_score_fn,\
                    attn_hist_construct_fn = attention_decoder_fn.prepare_attention(
                            hist_output, 'luong', num_units)

        attn_keys = (attn_encoder_keys, attn_hist_keys)
        attn_values = (attn_encoder_values, attn_hist_values)
        def attn_score_fn(attention_query, attention_keys, attention_values):
            pass
        def attn_construct_fn(attention_query, attention_keys, attention_values):
            attn_encoder = attn_encoder_construct_fn(attention_query,
                    attention_keys[0], attention_values[0])
            attn_hist = attn_hist_construct_fn(attention_query,
                    attention_keys[1], attention_values[1])
            return attn_encoder + tf.multiply(attn_hist, gate)

        # get output projection function
        output_fn, sampled_sequence_loss = output_projection_layer(num_units,
                num_symbols, num_samples)

        # get decoding loop function
        decoder_fn_train = attention_decoder_fn.attention_decoder_fn_train(encoder_state,
                attn_keys, attn_values, attn_score_fn, attn_construct_fn)
        decoder_fn_inference = attention_decoder_fn.attention_decoder_fn_inference(output_fn,
                encoder_state, attn_keys, attn_values, attn_score_fn,
                attn_construct_fn, self.embed, GO_ID, EOS_ID, max_length, num_symbols)

        # rnn decoder for generation
        with tf.variable_scope('decoder'):
            decoder_distribution, _, _ = dynamic_rnn_decoder(cell, decoder_fn_inference)
        generation_index = tf.argmax(tf.split(decoder_distribution,
            [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
        self.generation = tf.nn.embedding_lookup(self.symbols, generation_index)

        # rnn decoder for training
        with tf.variable_scope('decoder', reuse=True):
            decoder_output, _, _ = dynamic_rnn_decoder(cell, decoder_fn_train,
                    decoder_input, self.inputs['resp_len'])

        # calculate the loss of decoder
        self.loss['decoder'] = sampled_sequence_loss(decoder_output, resp, resp_mask)

        # building graph finished and get all parameters
        self.params = tf.trainable_variables()

        # calculate the gradient of parameters
        self.gradient_norm = {}
        self.update = {}
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        for key in self.loss:
            gradients = tf.gradients(self.loss[key], self.params)
            clipped_gradients, self.gradient_norm[key] = tf.clip_by_global_norm(
                    gradients, max_gradient_norm)
            self.update[key] = opt.apply_gradients(zip(clipped_gradients, self.params),
                    global_step=self.global_step)

        # build the saver
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, target, data, forward_only=False):
        input_feed = {self.inputs[key]: data[key] for key in data}
        if forward_only:
            output_feed = [self.loss[target]]
        else:
            output_feed = [self.loss[target], self.gradient_norm[target], self.update[target]]
        return session.run(output_feed, input_feed)

    def inference(self, session, data):
        input_feed = {self.inputs[key]: data[key] for key in data}
        output_feed = [self.generation]
        return session.run(output_feed, input_feed)

