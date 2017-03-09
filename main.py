import numpy as np
import tensorflow as tf
import sys
import time
import random
random.seed(1229)

from model import Seq2SeqModel, _START_VOCAB
try:
    from wordseg_python import Global
except:
    Global = None

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 40000, "vocabulary size.")
tf.app.flags.DEFINE_integer("embed_units", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, default isscreen")
tf.app.flags.DEFINE_integer("step_threshold", 200000, "")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    with open('%s/%s.post' % (path, fname)) as f:
        post = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.response' % (path, fname)) as f:
        response = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.keys' % (path, fname)) as f:
        key = [map(int, line.strip().split()) for line in f.readlines()]
    data = []
    for p, r, k in zip(post, response, key):
        data.append({'post': p, 'response': r, 'key': k})
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

def build_manual_data(data):
    cache = {}
    for item in data:
        key = item['key'][0]
        if key in cache:
            cache[key].append(item)
        else:
            cache[key] = [item]
    same, diff = [], []
    for item in data:
        key = item['key'][0]
        for other in cache[key]:
            if item != other:
                another = random.choice(data)
                same.append((item, other))
                diff.append((item, another))
    return same, diff

def padding(sent, l):
    return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

def gen_batched_data(data):
    batched_data = {}
    batched_data['post'] = [pair['post'] for pair, hist in data]
    batched_data['resp'] = [pair['response'] for pair, hist in data]
    batched_data['hist_post'] = [hist.get('post', []) for pair, hist in data]
    batched_data['hist_resp'] = [hist.get('response', []) for pair, hist in data]
    batched_data['gate'] = [hist != {} for pair, hist in data]

    for key in ['post', 'resp', 'hist_post', 'hist_resp']:
        batched_data[key+'_len'] = [len(item)+1 for item in batched_data[key]]
        length = max(batched_data[key+'_len'])
        batched_data[key] = np.array([padding(item, length) for item in batched_data[key]])

    return batched_data

def get_train_rate(step):
    if step < FLAGS.step_threshold:
        return 0
    if step < FLAGS.step_threshold * 2:
        return (step-FLAGS.step_threshold) / 2.0 / FLAGS.step_threshold
    return 0.5

def train(model, sess, data_train, manual_train_same):
    selected_data = []
    for i in range(FLAGS.batch_size):
        if random.random() < get_train_rate(model.global_step.eval()):
            selected_data.append(random.choice(manual_train_same))
        else:
            selected_data.append((random.choice(data_train), {}))
    batched_data = gen_batched_data(selected_data)
    outputs = model.step_decoder(sess, 'decoder', batched_data)
    return outputs[0]

def evaluate(model, sess, data, name):
    loss = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data):
        selected_data = data[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, 'decoder', batched_data, forward_only=True)
        loss += outputs[0]
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    print('    perplexity on dev set (%s): %.2f' % (name, np.exp(loss)))

def inference(model, sess, post, hist_post, hist_resp, gate):
    batched_data = {}
    batched_data['post'] = post
    batched_data['hist_post'] = hist_post
    batched_data['hist_resp'] = hist_resp
    batched_data['gate'] = gate

    for key in ['post', 'hist_post', 'hist_resp']:
        batched_data[key+'_len'] = [len(item)+1 for item in batched_data[key]]
        length = max(batched_data[key+'_len'])
        batched_data[key] = np.array([padding(item, length) for item in batched_data[key]])

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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        data_train = load_data(FLAGS.data_dir, 'weibo_pair_train')
        data_dev = load_data(FLAGS.data_dir, 'weibo_pair_dev')
        manual_train = load_data(FLAGS.data_dir, 'manual_train')
        manual_dev = load_data(FLAGS.data_dir, 'manual_dev')
        vocab, embed = build_vocab(FLAGS.data_dir, data_train + manual_train*10)
        manual_train_same, manual_train_diff = build_manual_data(manual_train)
        manual_dev_same, manual_dev_diff = build_manual_data(manual_dev)

        model = Seq2SeqModel(
                FLAGS.symbols,
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
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
                evaluate(model, sess, zip(data_dev, [{}]*len(data_dev)), 'weibo')
                evaluate(model, sess, manual_dev_same, 'manual')
                if np.sum(loss_step) > max(previous_losses):
                    sess.run(model.learning_rate_decay_op)
                previous_losses = previous_losses[1:]+[np.sum(loss_step)]
                loss_step, time_step = np.zeros((1, )), .0

            start_time = time.time()
            loss_step += train(model, sess, data_train, manual_train_same) / FLAGS.per_checkpoint
            time_step += (time.time() - start_time) / FLAGS.per_checkpoint

    else:
        model = Seq2SeqModel(
                FLAGS.symbols,
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                vocab=None)
        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)
        model.symbol2index.init.run()

        def split(sent):
            if Global == None:
                return sent.split()

            sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
            tuples = [(word.decode("gbk").encode("utf-8"), pos)
                    for word, pos in Global.GetTokenPos(sent)]
            return [each[0] for each in tuples]

        if FLAGS.inference_path == '':
            while True:
                sys.stdout.write('post history: ')
                sys.stdout.flush()
                hist_post = split(sys.stdin.readline().strip())
                sys.stdout.write('response history: ')
                sys.stdout.flush()
                hist_resp = split(sys.stdin.readline().strip())
                sys.stdout.write('new post: ')
                sys.stdout.flush()
                post = split(sys.stdin.readline().strip())
                gate = len(hist_post+hist_resp) > 0
                response = inference(model, sess, [post], [hist_post], [hist_resp], [gate])[0]
                print('response: %s' % ''.join(response))
                sys.stdout.flush()
        '''
        else:
            posts = []
            with open(FLAGS.inference_path) as f:
                for line in f:
                    sent = line.strip().split('\t')[0]
                    posts.append(split(sent))

            responses = []
            st, ed = 0, FLAGS.batch_size
            while st < len(posts):
                responses += inference(model, sess, posts[st: ed])
                st, ed = ed, ed+FLAGS.batch_size

            with open(FLAGS.inference_path+'.out', 'w') as f:
                for p, r in zip(posts, responses):
                    #f.writelines('%s\t%s\n' % (''.join(p), ''.join(r)))
                    f.writelines('%s\n' % (''.join(r)))
        '''


