# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function

import tensorflow.python.platform

import collections
import math
import numpy as np
import os
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import zipfile

# Jadd
NUM_CORES = 8
# tf.ConfigProto( 
#   allow_soft_placement=FLAGS.allow_soft_placement, 
#   log_device_placement=FLAGS.log_device_placement, 
#   inter_op_parallelism_threads=NUM_CORES, 
#   intra_op_parallelism_threads=NUM_CORES)
session_conf = tf.ConfigProto( 
  allow_soft_placement=True, 
  log_device_placement=True, 
  inter_op_parallelism_threads=NUM_CORES, 
  intra_op_parallelism_threads=NUM_CORES) 


# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a string.
def read_data(filename):
  """
  [TODO] Not sure how zipfile reads

  Output:
    words: (list) word strings, e.g. ['originated', 'as', 'a', 'term']
  """
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return f.read(name).split()
  f.close()

words = read_data(filename)
print('Data size', len(words))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words):
  """
  Input: 
    words: see read_data()

  Output:
    data: (list) 'words' converted into indices
    count: (list) word counts, e.g. [(UNK, 1537), ('Apple', 10), ...]
    dictionary: (dict) word-to-index mapping, e.g. {'Apple': 1, 'UNK': 50001, ...}
    reverse_dictionary: (dict) index-to-word mapping.
  """
  count = [['UNK', -1]]
  count.extend(
    collections.Counter(words).most_common(
      vocabulary_size - 1))
  # J: Pydoc: "extend list by appending elements from the iterable"
  #    which is a more handy way!
  #    The input of extend() is an iterable.
  #    So what's the input? (collections.Counter)
  #    Pydoc: c = Counter('abcdabcaba')
  #           c = [('a', 4), ('b', 3), ('c', 2), ('d', 1)]
  #           which is a list of tuples
  #    Note that Counter is a class, and the most commonly used 
  #    method of Counter is most_common()

  # Word Indexing
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)

  # Because vocabulary size is pre-defined, the OOVs
  # are assigned to 'UNK' which is the first word of this dictionary
  # *The 1st element of count is 'UNK'
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)

  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])

# data_index is a cursor to the whole data (which is represented
# by a stream of indices of words.
# For illustration:
#   words: This is a book that ...
#    data: 5810  9 2  512   30 ...
data_index = 0  


# Step 4: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  '''
  Input:
     batch_size: #samples of a batch
      num_skips: #times a word appear in a batch
    skip_window: #words forward/backwards
  
  Output: 
    batch
    label

  This function generates skip-gram training pairs
  Skip-gram (skipped word is enclosed in quotation marks):
    [anarchism 'originated' as]
    [originated 'as' a]
    [as 'a' term]
  The pseudo-pairs should be converted into pairs (two-tuple), i.e.
    [anarchism 'originated' as] => (originated, anarchism), (originated, as)
    [originated 'as' a]         => (as, originated), (as, a)
    [as 'a' term]               => (a, as), (a, term)
  
  **Skip-gram may be conceptually sound, 
    but this detail-- turning a gram into pairs
    is crucial when implementing.

  Note:
  data: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']
  
  with num_skips = 2 and skip_window = 1:
      batch: ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']
      labels: ['as', 'anarchism', 'a', 'originated', 'term', 'as', 'a', 'of']
  
  with num_skips = 4 and skip_window = 2:
      batch: ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a']
      labels: ['anarchism', 'originated', 'term', 'a', 'as', 'of', 'originated', 'term']
  '''
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)   # J: index representation 
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # So now buffer = [13 25 995 4063 7] := ['originated', 'from', ...]

  # J this passage is difficult.
  # Note: // is integer division
  # span = 5
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    # Note: skipped window is the target!
    targets_to_avoid = [ skip_window ]   # just an initial
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)  # role a dice to choose a word other than the center word.
      targets_to_avoid.append(target)
      batch[i * num_skips + j]     = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)   # update data_index
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

# Step 5: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
num_sampled = 64    # Number of negative examples to sample.


'''
The whole package of code cannot be execute properly 
without this function.
'''
def device_for_node(n):
  if n.type == "MatMul":
    return "/gpu:0"
  else:
    return "/cpu:0"


# Graph is an important part in Tensorflow.
# Make it a habit using it explicitly. 
# (enclosing every computation-graph assignment in a graph)
graph = tf.Graph()
with graph.as_default():
  with graph.device(device_for_node):
    # Input data.
    train_inputs  = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels  = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
    # Construct the variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
  
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, vocabulary_size))
  
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)


# Step 6: Begin training
num_steps = 100001
with tf.Session(graph=graph, config=session_conf) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    # Print out avg loss per 2000 batch
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


# Step 7: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)


try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)
except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")

# 
'''
An alternative to Word2Vec is called CBOW (Continuous Bag of Words). 
In the CBOW model, instead of predicting a context word from a word vector, 
you predict a word from the sum of all the word vectors in its context. 
Implement and evaluate a CBOW model trained on the text8 dataset.
'''
