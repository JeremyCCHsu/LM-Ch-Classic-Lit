# coding=utf-8

# [TODO] I'll replace 'UNK' with u'u\3012' (full-size post-office sign)
# [TODO] Current system over-trains 
#        (in less than 7000 steps if embedding is tunable)!
#        (inside perplexity: <20,  validation perplexity: ~350)

# This simplest model ahieves perplexity of less than 200 in 6000 steps!

# These are all the modules we'll be using later. 
# Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

import io
import cPickle
import sys
import math
import datetime

iFile = 'TangPoemsUTF8-5.txt'
# iFile = 'Zizhitongjan001-294.txt'
iModel = 'models/Zizhitongjan001-294-skip1-2016-0227.mdl'

batch_size = 64
num_unrollings = 8
valid_size = 50
num_nodes = 128   # hidden nodes


# [TODO] Should I use common words for validation? instead of the first piece?
def load_word2vec_model(filename):
  with open(filename, 'rb') as f:
    model = cPickle.load(f)
    W_w2v = model['W']
    b_w2v = model['b']
    word2index = model['word2index']
    index2word = model['index2word']
    wordcount = model['wordcount']
    embeddings = model['embedding']
    model = None
    return W_w2v, b_w2v, word2index, index2word, wordcount, embeddings

def read_data(filename):
  with io.open(filename,'r',encoding='utf8') as f:
      return f.read()

W_w2v, b_w2v, word2index, index2word, wordcount, embeddings = load_word2vec_model(iModel)
vocabulary_size, wordVecSize = embeddings.shape
# Replace 'UNK' token
unk = u'〒'
word2index[unk] = 0
index2word[0] = unk

text = read_data(iFile)
print('Data size %d' % len(text))


valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


# vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
# first_letter = ord(string.ascii_lowercase[0])
def char2id(char):
  return word2index.get(char, word2index['UNK'])

def id2char(dictid):
  return index2word[dictid]


batch_size = 64
num_unrollings = 10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
    """
    tz = 1000
    vz = 5
    bz = 10
    seg= 100
    cursor = [0, 100, 200, 300, ..., 900] 1xbz
    """
  def _next_batch(self):
    """
    Generate a single batch from the current cursor position in the data.
    J: A batch is a batch_size x voca_size sparse matrix
    batch.shape: 10 x 5
    batch[b=0, a] = 1;    # a = char2id(self._text[self._cursor[b]])
                          #    _cursor[b=0] = 0;
    batch[b=1, a] = 1;    # a = char2id(self._text[self._cursor[b]])
                          #    _cursor[b=1] = 100;

    cursor = [1, 101, 201, 301, ..0, 901] 1xbz
    J: So batch is not a consecutive seq? why?
       The answer is that the variable name 'batch' is really confusing.
       Here, a 'batch' is actually a piece of a batch
       Later in the 'next' method, we can see how these pieces are combined,
       but it's another weired process.
    """
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.int32)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      # batch[b, 0] = char2id(self._text[self._cursor[b]])
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  def next(self):
    """
    Generate the next array of batches from the data. 
    The array consists of the last batch of the previous array, 
    followed by num_unrollings new ones.

    J: a batch is batch_sz x voc_sz
       'bactchs' is (num_unrolling x batch_sz x voc_sz), which is 3D!
       Think of it as a cube with axes: z=n, x=b, y=v
       so a slice of yz-plane is a sample (in a batch).
    J: You should think n-dimensionally instead of keeping Matlab (2D) habits.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches


def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """
  Convert a sequence of batches back into their (most likely) string
  representation.
  J: this method is baffling (however clever).
  """
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

# # print(batches2string(train_batches.next()))
# print(batches2string(train_batches.next()))
# print(batches2string(valid_batches.next()))
# print(batches2string(valid_batches.next()))


def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """
  Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  J: roll a threshold (0, 1)
     pick the dim (which is the character) whose CDF is just larger 
     than the threshold.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """
  Turn a (column) prediction into 1-hot encoded samples.
  J: Generate one character (in the form of 1-of-K coding).
  """
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]





num_nodes = 128
graph = tf.Graph()
with graph.as_default(): 
  # Parameters:
  # Input gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([wordVecSize, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([wordVecSize, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
  cx = tf.Variable(tf.truncated_normal([wordVecSize, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([wordVecSize, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state  = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  # E = tf.constant(embeddings)   # [TODO] Variable/Const: with/without embedding update
  E = tf.Variable(embeddings)   # [TODO] Variable/Const: with/without embedding update
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    # i = tf.nn.embedding_lookup(embeddings, i)
    i = tf.matmul(i, E)
    input_gate  = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state  = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state
  # J: Why is it that 'train_data' and 'outputs' stored in a list?
  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    # train_data.append(
    #   tf.placeholder(tf.int32, shape=[batch_size]))
    train_data.append(
      tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.
  # Unrolled LSTM loop.
  outputs = list()
  output  = saved_output
  state   = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)
  # State saving across unrollings. ([TODO] J: WTF is this?)
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    # y = list()
    # for unroll in train_labels:
    #   y_ = np.zeros((batch_size, vocabulary_size))
    #   for i in range(batch_size):
    #     y_[i, unroll[i]] = 1.0
    #   y.append(y_)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))
  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)
  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


# 
fid = open('TangPoem.log', 'w')
num_steps = 10001
summary_frequency = 100
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(96):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
          fid.write(sentence.encode('utf8'))
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        batch = valid_batches.next()
        predictions = sample_prediction.eval({sample_input:  batch[0]})
        valid_logprob = valid_logprob + logprob(predictions, batch[1])
      print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))

  model = dict()
  model['ix'] = session.run(ix)
  model['im'] = session.run(im)
  model['ib'] = session.run(ib)
  model['fx'] = session.run(fx)
  model['fm'] = session.run(fm)
  model['fb'] = session.run(fb)
  model['cx'] = session.run(cx)
  model['cm'] = session.run(cm)
  model['cb'] = session.run(cb)
  model['ox'] = session.run(ox)
  model['om'] = session.run(om)
  model['ob'] = session.run(ob)
  model['saved_output'] = session.run(saved_output)
  model['saved_state '] = session.run(saved_state)
  model['w'] = session.run(w)
  model['b'] = session.run(b)
  model['E'] = session.run(E)
  timestamp = datetime.datetime.now().strftime('%Y-%m%d-%H%M')
  oFile = 'models/' + iFile.replace('.txt', '-%s.mdl' % timestamp)
  with open(oFile, 'wb') as f:
    cPickle.dump(model, f)


# 
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  # To show UTF-8 Chinese glyphs, these 2 lines are needed
  from matplotlib.font_manager import FontProperties
  prop = FontProperties()
  prop.set_file('wt064.ttf')    # [TODO] Specify a font (or at least solve this problem more elegantly)
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom',
                 fontproperties=prop)
  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(model['E'][:plot_only,:])
  labels = [index2word[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)
except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")

# 


'''
Problem 1
You might have noticed that the definition of the LSTM cell involves 
4 matrix multiplications with the input, 
and 4 matrix multiplications with the output. 
Simplify the expression by using a single matrix multiply for each, 
and variables that are 4 times larger.


Problem 2
We want to train a LSTM over bigrams, that is pairs of consecutive 
characters like 'ab' instead of single characters like 'a'. 
Since the number of possible bigrams is large, feeding them directly to 
the LSTM using 1-hot encodings will lead to a very sparse representation 
that is very wasteful computationally.
  a- Introduce an embedding lookup on the inputs, 
     and feed the embeddings to the LSTM cell instead of the inputs 
     themselves.
  b- Write a bigram-based LSTM, modeled on the character LSTM above.
     http://arxiv.org/abs/1409.2329
  c- Introduce Dropout. 
     For best practices on how to use Dropout in LSTMs, 
     refer to this article.
     http://arxiv.org/abs/1409.3215

Problem 3 (difficult!)
Write a sequence-to-sequence LSTM which mirrors all the words in a sentence. 
For example, if your input is:
    the quick brown fox

the model should attempt to output:
    eht kciuq nworb xof

Refer to the lecture on how to put together a sequence-to-sequence model, as well as this article for best practices.
'''



'''
TensorFlow does have a limit of 2GB on the GraphDef protos, which stems from a limitation of the protocol buffers implementation. You can quickly reach that limit if you have large constant tensors in your graph. In particular, if you use the same numpy array multiple times, TensorFlow will add multiple constant tensors to your graph.

In your case, mnist.train.images returned by input_data.read_data_sets is a numpy floating point array with shape (55000, 784), so it is about 164 MB. You pass that numpy array to rbm_layer.cd1, and inside that function, every time you use visibles, a TensorFlow Const node gets created from the numpy array. You use visibiles in 3 locations, so every call to cd1 is increasing the graph size by approximately 492 MB, so you easily exceed the limit. The solution is to create a TensorFlow constant once and pass that constant to the cd1 function like so :

trX_constant = tf.constant(trX)
for i in range(10):
    print "RBM CD: ", i
    rbm_layer.cd1(trX_constant)

BTW, I am not sure what your intention is in the above loop. 
Note that the cd1 function simply adds the assign_add nodes to the graph, 
and does NOT actually perform the assigns. 
If you really want those assigns to happen while you train, 
you should consider chaining those assigns via control dependencies to your final train_op node.
'''