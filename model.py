import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn
from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
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
            is_train,
            vocab=None,
            embed=None,
            learning_rate=0.5,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=False):

        # build the vocab table (string to index)
        if is_train:
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

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate),
                trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        def build_decoder():
            placeholder, loss, output = None, None, None

            posts = tf.placeholder(tf.string, shape=(None, None))  # batch*len
            posts_length = tf.placeholder(tf.int32, shape=(None))  # batch
            responses = tf.placeholder(tf.string, shape=(None, None))  # batch*len
            responses_length = tf.placeholder(tf.int32, shape=(None))  # batch
            placeholder = {'posts': posts, 'posts_length': posts_length,
                    'responses': responses, 'responses_length': responses_length}

            posts_input = self.symbol2index.lookup(posts)   # batch*len
            responses_target = self.symbol2index.lookup(responses)   #batch*len

            batch_size, decoder_len = tf.shape(responses)[0], tf.shape(responses)[1]
            responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
                tf.split(responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch*len
            decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(responses_length-1,
                decoder_len), reverse=True, axis=1), [-1, decoder_len])

            encoder_input = tf.nn.embedding_lookup(self.embed, posts_input) #batch*len*unit
            decoder_input = tf.nn.embedding_lookup(self.embed, responses_input)

            if use_lstm:
                cell = MultiRNNCell([LSTMCell(num_units)] * num_layers)
            else:
                cell = MultiRNNCell([GRUCell(num_units)] * num_layers)

            # rnn encoder
            encoder_output, encoder_state = dynamic_rnn(cell, encoder_input,
                    posts_length, dtype=tf.float32, scope="encoder")

            # get output projection function
            output_fn, sampled_sequence_loss = output_projection_layer(num_units,
                    num_symbols, num_samples)

            # get attention function
            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                    = attention_decoder_fn.prepare_attention(encoder_output, 'luong', num_units)

            # get decoding loop function
            decoder_fn_train = attention_decoder_fn.attention_decoder_fn_train(encoder_state,
                    attention_keys, attention_values, attention_score_fn, attention_construct_fn)
            decoder_fn_inference = attention_decoder_fn.attention_decoder_fn_inference(output_fn,
                    encoder_state, attention_keys, attention_values, attention_score_fn,
                    attention_construct_fn, self.embed, GO_ID, EOS_ID, max_length, num_symbols)

            if is_train:
                # rnn decoder
                decoder_output, _, _ = dynamic_rnn_decoder(cell, decoder_fn_train,
                        decoder_input, responses_length, scope="decoder")
                # calculate the loss of decoder
                loss = sampled_sequence_loss(decoder_output, responses_target, decoder_mask)
            else:
                # rnn decoder
                decoder_distribution, _, _ = dynamic_rnn_decoder(cell, decoder_fn_inference,
                        scope="decoder")

                # generating the response
                #generation_index = tf.argmax(decoder_distribution, 2)
                generation_index = tf.argmax(tf.split(decoder_distribution,
                    [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
                output = tf.nn.embedding_lookup(self.symbols, generation_index)

            return placeholder, loss, output

        def build_matcher():
            unit = 256
            placeholder, loss, output = None, None, None

            posts = tf.placeholder(tf.string, shape=(2, None, None))  # 2*batch*len
            posts_length = tf.placeholder(tf.int32, shape=(2, None))  # 2*batch
            solution = tf.placeholder(tf.float32, shape=(None))  # batch

            placeholder = {'posts': posts, 'posts_length': posts_length, 'solution': solution}

            batch_size, post_len = tf.shape(posts)[1], tf.shape(posts)[2]

            posts_input = self.symbol2index.lookup(posts)  # 2*batch*len
            encoder_input = tf.nn.embedding_lookup(self.embed, posts_input)  # 2*batch*len*unit

            encoder_input = tf.reshape(encoder_input,
                    [2*batch_size, post_len, num_embed_units])  # 2batch*len*unit
            posts_length = tf.reshape(posts_length, [2*batch_size]) # 2batch

            cell = GRUCell(unit)
            encoder_output, encoder_state = dynamic_rnn(cell, encoder_input,
                    posts_length, dtype=tf.float32, scope="matcher_encoder")
            # encoder_output is 2batch*len*unit, encoder_state is 2batch*unit

            [state_a, state_b] = tf.split(encoder_state, [batch_size, batch_size], 0)
            state_a = tf.reshape(state_a, [batch_size, unit])
            state_b = tf.reshape(state_b, [batch_size, unit])

            state_mid = tf.reshape(layers.linear(state_a, unit), [batch_size*unit])
            state_b = tf.reshape(state_b, [batch_size*unit])
            similarity = tf.reduce_sum(tf.reshape(state_mid*state_b, [batch_size, unit]), 1)

            output = tf.sigmoid(similarity)

            if is_train:
                loss = -tf.log(output+1e-12)*solution -tf.log(1-output+1e-12)*(1-solution)
                loss = tf.reduce_sum(loss) / tf.cast(batch_size, tf.float32)

            return placeholder, loss, output

        self.placeholder = {}
        self.loss = {}
        self.output = {}
        self.placeholder['decoder'], self.loss['decoder'], \
                self.output['decoder'] = build_decoder()
        self.placeholder['matcher'], self.loss['matcher'], \
                self.output['matcher'] = build_matcher()

        # build graph finished and get all parameters
        self.params = tf.trainable_variables()
        if is_train:
            self.gradient_norm = {}
            self.update = {}
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for target in ['decoder', 'matcher']:
                # calculate the gradient of parameters
                gradients = tf.gradients(self.loss[target], self.params)
                clipped_gradients, self.gradient_norm[target] = tf.clip_by_global_norm(
                        gradients, max_gradient_norm)
                self.update[target] = opt.apply_gradients(
                        zip(clipped_gradients, self.params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step(self, session, target, data, forward_only=False):
        input_feed = {self.placeholder[target][key]: data[key] for key in data}
        if forward_only:
            output_feed = [self.loss[target]]
        else:
            output_feed = [self.loss[target], self.gradient_norm[target], self.update[target]]
        return session.run(output_feed, input_feed)

    def match(self, session, data):
        input_feed = {self.placeholder['matcher'][key]: data[key] for key in data}
        output_feed = [self.output['matcher']]
        return session.run(output_feed, input_feed)

    def inference(self, session, data):
        input_feed = {self.placeholder['decoder'][key]: data[key] for key in data}
        output_feed = [self.output['decoder']]
        return session.run(output_feed, input_feed)

