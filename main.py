import numpy as np
import tensorflow as tf
import sys
import os
import time
import random
random.seed(1229)

from model import Seq2SeqModel, _START_VOCAB
from wordseg_python import Global
from rerank import rerank

sys.path.append('reranker/')
from reranker import reranker, score

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 40000, "vocabulary size.")
tf.app.flags.DEFINE_integer("embed_units", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("beam_size", 5, "Beam size to use during beam inference.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_inpath", "", "Set filename of inference, default isscreen")
tf.app.flags.DEFINE_string("inference_outpath", "", "Set filename of inference output")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    with open('%s/%s.post' % (path, fname)) as f:
        post = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.response' % (path, fname)) as f:
        response = [line.strip().split() for line in f.readlines()]
    data = []
    for p, r in zip(post, response):
        data.append({'post': p, 'response': r})
    return data

def build_vocab(path, data):
    print("Creating vocabulary...")
    vocab = {}
    for i, pair in enumerate(data):
        if i % 100000 == 0:
            print("    processing line %d" % i)
        for token in pair['post']+pair['response']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]

    print("Loading word vectors...")
    vectors = {}
    with open('%s/vector.txt' % path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector

    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float, vectors[word].split())
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)

    return vocab_list, embed

def gen_batched_data(data):
    encoder_len = max([len(item['post']) for item in data])+1
    decoder_len = max([len(item['response']) for item in data])+1

    posts, responses, posts_length, responses_length = [], [], [], []
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post'])+1)
        responses_length.append(len(item['response'])+1)

    batched_data = {'posts': np.array(posts),
            'responses': np.array(responses),
            'posts_length': posts_length,
            'responses_length': responses_length}
    return batched_data

def train(model, sess, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = gen_batched_data(selected_data)
    outputs = model.step_decoder(sess, batched_data)
    return outputs[0]

def evaluate(model, sess, data_dev):
    loss = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, forward_only=True)
        loss += outputs[0]
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    print('    perplexity on dev set: %.2f' % np.exp(loss))

def inference(model, sess, posts):
    length = [len(p)+1 for p in posts]
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
    batched_posts = [padding(p, max(length)) for p in posts]
    batched_data = {'posts': np.array(batched_posts),
            'posts_length': np.array(length, dtype=np.int32)}
    responses = model.inference(sess, batched_data)[0]
    results = []
    for response in responses:
        result = []
        for token in response:
            if token != '_EOS':
                result.append(token)
            else:
                break
        results.append(result)
    return results

def beam_inference(model, sess, posts):
    length = [len(p)+1 for p in posts]
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
    batched_posts = [padding(p, max(length)) for p in posts]
    batched_data = {'posts': np.array(batched_posts),
            'posts_length': np.array(length, dtype=np.int32)}
    responses = model.beam_inference(sess, batched_data)
    results = []
    for response, prb in responses:
        result = []
        for token in response:
            if token != '_EOS':
                result.append(token)
            else:
                break
        results.append((result, prb))
    return results

def get_model_list(path):
    fname_list = os.listdir(path)
    f_list = []
    for fname in fname_list:
        if fname.find('.index') >= 0:
            version = int(fname[11:19])
            if (300000 < version) & (version < 600000):
                f_list.append(path+'/'+fname.replace('.index', ''))
    f_list.sort()
    return f_list[20:21]

def get_posts(path):
    def split(sent):
        sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
        tuples = [(word.decode("gbk").encode("utf-8"), pos)
                for word, pos in Global.GetTokenPos(sent)]
        return [each[0] for each in tuples]
    with open(path) as f:
        posts = [split(line.strip()) for line in f]
    return posts

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        data_train = load_data(FLAGS.data_dir, 'weibo_pair_train')
        data_dev = load_data(FLAGS.data_dir, 'weibo_pair_dev')
        vocab, embed = build_vocab(FLAGS.data_dir, data_train)

        model = Seq2SeqModel(
                FLAGS.symbols,
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                embed=embed)
        if FLAGS.log_parameters:
            model.print_parameters()

        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            model.symbol2index.init.run()
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            model.symbol2index.init.run()

        loss_step, time_step = np.zeros((1, )), .0
        previous_losses = [1e18]*3
        while True:
            if model.global_step.eval() % FLAGS.per_checkpoint == 0:
                show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                print("global step %d learning rate %.4f step-time %.2f perplexity %s"
                        % (model.global_step.eval(), model.learning_rate.eval(),
                            time_step, show(np.exp(loss_step))))
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir,
                        global_step=model.global_step)
                evaluate(model, sess, data_dev)
                if np.sum(loss_step) > max(previous_losses):
                    sess.run(model.learning_rate_decay_op)
                previous_losses = previous_losses[1:]+[np.sum(loss_step)]
                loss_step, time_step = np.zeros((1, )), .0

            start_time = time.time()
            loss_step += train(model, sess, data_train) / FLAGS.per_checkpoint
            time_step += (time.time() - start_time) / FLAGS.per_checkpoint

    else:
        data_dev = load_data(FLAGS.data_dir, 'weibo_pair_dev')
        model = Seq2SeqModel(
                FLAGS.symbols,
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                beam_size=FLAGS.beam_size,
                remove_unk=True,
                d_rate=0.0,
                is_train=False,
                vocab=None)
        model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        model.symbol2index.init.run()
        model_list = get_model_list(FLAGS.train_dir)


        posts = get_posts(FLAGS.inference_inpath)
        responses = [[] for _ in range(len(posts))]
	with tf.Session(config=config) as _sess:
	    mmi = reranker(_sess, 'reranker/model')
            for model_path in model_list:
                print('restore from %s' % model_path)
                model.saver.restore(sess, model_path)
                #evaluate(model, sess, data_dev)
                for i, post in enumerate(posts):
                    responses[i] += beam_inference(model, sess, [post]*FLAGS.beam_size)
            results = []
            for i in range(len(posts)):
                items = rerank(posts[i], responses[i])
                mmiscore = score(_sess, mmi, [posts[i],[item[0] for item in items]])
                _items = []
                for idx, item in enumerate(items):
                    _items.append(tuple([item[0], mmiscore[idx], item[1], item[2]]))
                items = _items
                items.sort(key=lambda x:x[3]*0.01+x[2]+x[1])
                results.append(items)
            with open(FLAGS.inference_outpath+'.log', 'w') as f:
                for post, items in zip(posts, results):
                    f.writelines(' '.join(post)+'\n')
                    for item in items:
                        f.writelines('%.2f\t%.2f\t%.2f:\t%s\n' % (item[3], item[2], item[1], ' '.join(item[0])))
                    f.writelines('-'*50+'\n')
            with open(FLAGS.inference_outpath, 'w') as f:
                for post, items in zip(posts, results):
                    items.sort(key=lambda x:x[3]*0.01+x[2]+x[1])
                    f.writelines('%s\t%s\n' % (''.join(post), ''.join(items[0][0])))
                    #items.sort(key=lambda x:x[3]*0.01+x[1])
                    #f.writelines('%s\t%s\n' % (''.join(post), ''.join(items[0][0])))
                    #items.sort(key=lambda x:x[3]*0.01+x[2])
                    #f.writelines('%s\t%s\n' % (''.join(post), ''.join(items[0][0])))
