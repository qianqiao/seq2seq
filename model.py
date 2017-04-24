import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn 
from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from output_projection import output_projection_layer
from beam_inference import attention_decoder_fn_beam_inference
from tensorflow.contrib.session_bundle import exporter

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
            beam_size,
            embed,
            learning_rate=0.5,
            remove_unk=False,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=8,
            use_lstm=False):
        
        self.posts = tf.placeholder(tf.string, (None, None), 'enc_inps')  # batch*len
        self.posts_length = tf.placeholder(tf.int32, (None), 'enc_lens')  # batch
        self.responses = tf.placeholder(tf.string, (None, None), 'dec_inps')  # batch*len
        self.responses_length = tf.placeholder(tf.int32, (None), 'dec_lens')  # batch
        
        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), 
                trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=UNK_ID,
                shared_name="in_table",
                name="in_table",
                checkpoint=True)
        self.index2symbol = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_UNK',
                shared_name="out_table",
                name="out_table",
                checkpoint=True)
        # build the vocab table (string to index)

        self.posts_input = self.symbol2index.lookup(self.posts)   # batch*len
        self.responses_target = self.symbol2index.lookup(self.responses)   #batch*len
        
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int64)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch*len
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1, 
            decoder_len), reverse=True, axis=1), [-1, decoder_len])
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        
        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts_input) #batch*len*unit
        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input) 

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units)] * num_layers)
        else:
            cell = MultiRNNCell([GRUCell(num_units)] * num_layers)
        
        # rnn encoder
        encoder_output, encoder_state = dynamic_rnn(cell, self.encoder_input, 
                self.posts_length, dtype=tf.float32, scope="encoder")

        # get output projection function
        output_fn, sampled_sequence_loss = output_projection_layer(num_units, 
                num_symbols, num_samples)

        # get attention function
        attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                = attention_decoder_fn.prepare_attention(encoder_output, 'luong', num_units)
        

        with tf.variable_scope('decoder'):
            decoder_fn_train = attention_decoder_fn.attention_decoder_fn_train(
                    encoder_state, attention_keys, attention_values,
                    attention_score_fn, attention_construct_fn)
            self.decoder_output, _, _ = dynamic_rnn_decoder(cell, decoder_fn_train, 
                    self.decoder_input, self.responses_length, scope="decoder_rnn")
            self.decoder_loss = sampled_sequence_loss(self.decoder_output, 
                    self.responses_target, self.decoder_mask)
        
        with tf.variable_scope('decoder', reuse=True):
            decoder_fn_inference = attention_decoder_fn.attention_decoder_fn_inference(
                    output_fn, encoder_state, attention_keys, attention_values, 
                    attention_score_fn, attention_construct_fn, self.embed, GO_ID, 
                    EOS_ID, max_length, num_symbols)
                
            self.decoder_distribution, _, _ = dynamic_rnn_decoder(cell,
                    decoder_fn_inference, scope="decoder_rnn")
            self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
                [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
            self.generation = self.index2symbol.lookup(self.generation_index, name='generation') 
        
        with tf.variable_scope('decoder', reuse=True):
            decoder_fn_beam_inference = attention_decoder_fn_beam_inference(output_fn,
                    encoder_state, attention_keys, attention_values, attention_score_fn,
                    attention_construct_fn, self.embed, GO_ID, EOS_ID, max_length,
                    num_symbols, beam_size, remove_unk)
            _, _, self.context_state = dynamic_rnn_decoder(cell, 
                    decoder_fn_beam_inference, scope="decoder_rnn")
            (log_beam_probs, beam_parents, beam_symbols, result_probs, 
                    result_parents, result_symbols) = self.context_state

            self.beam_parents = tf.transpose(tf.reshape(beam_parents.stack(), 
                [max_length+1, -1, beam_size]), [1,0,2], name='beam_parents')
            self.beam_symbols = tf.transpose(tf.reshape(beam_symbols.stack(), 
                [max_length+1, -1, beam_size]), [1,0,2])
            self.beam_symbols = self.index2symbol.lookup(tf.cast(self.beam_symbols, 
                tf.int64), name="beam_symbols")

            self.result_probs = tf.transpose(tf.reshape(result_probs.stack(), 
                [max_length+1, -1, beam_size*2]), [1,0,2], name='result_probs')
            self.result_symbols = tf.transpose(tf.reshape(result_symbols.stack(), 
                [max_length+1, -1, beam_size*2]), [1,0,2])
            self.result_parents = tf.transpose(tf.reshape(result_parents.stack(), 
                [max_length+1, -1, beam_size*2]), [1,0,2], name='result_parents')
            self.result_symbols = self.index2symbol.lookup(tf.cast(self.result_symbols, 
                tf.int64), name='result_symbols')
        

        self.params = tf.trainable_variables()
            
        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        
        # Exporter for serving 
        self.model_exporter = exporter.Exporter(self.saver) 
        inputs = {"enc_inps:0": self.posts, "enc_lens:0":self.posts_length}
        outputs = {"beam_symbols":self.beam_symbols, "beam_parents":self.beam_parents, 
                "result_probs":self.result_probs, "result_symbols":self.result_symbols,
                "result_parents": self.result_parents}
        self.model_exporter.init(tf.get_default_graph().as_graph_def(), named_graph_signatures={ "inputs": exporter.generic_signature(inputs), "outputs": exporter.generic_signature(outputs)}) 


    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def step_decoder(self, session, data, forward_only=False):
        input_feed = {self.posts: data['posts'],
                self.posts_length: data['posts_length'],
                self.responses: data['responses'],
                self.responses_length: data['responses_length']}
        if forward_only:
            output_feed = [self.decoder_loss]
        else:
            output_feed = [self.decoder_loss, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
