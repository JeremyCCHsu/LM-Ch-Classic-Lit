'''
This toy RNN is a regressor to a 2D [cos, sin] function

The learning is very difficult.
Arbitrary learning rate is likely to fail! 

Result 1:
  Not very perfect, but it catches the trend (1, 0, -1, 0)
  tanh achieved better result than (tanh-hidden, linear-out)
  and (relu-hidden, linear-out).
  Training error increases after step 140.
  
  Regurlarization: 
    None (I tried L2, but it only made things worse)
  
  Optimizer:
    RMSPropOptimizer is better than GD. 
    Forgetting factor seems to be the large (1.0) the better.

'''

import math
import numpy as np
import tensorflow as tf
from __future__ import print_function


N = 1000
pi = math.pi
t = np.linspace(0, 1-1.0/N, N)
f = 250
dim = 2
x = np.zeros((N, dim))
x[:,0] = np.sin(2*pi*f*t)   # Nx1
x[:,1] = np.cos(2*pi*f*t)   # Nx1

num_unrollings = 5

batch_size = 100

# 3D representation: unroll batch
batch = list()
for i in xrange(num_unrollings+1):
    # batch.append( np.reshape(x[i:i+batch_size], [batch_size, dim]) )
    batch.append(x[i:i+batch_size]) 



# J: it's difficult to 'unroll'!

graph = tf.Graph()
with graph.as_default():
    # Wx = tf.Variable(np.identity(dim, dtype=np.float32))
    # Wy = tf.Variable(np.identity(dim, dtype=np.float32))
    # Wh = tf.Variable(np.identity(dim, dtype=np.float32))
    bx = tf.Variable(np.zeros((1, dim), dtype=np.float32))
    by = tf.Variable(np.zeros((1, dim), dtype=np.float32))
    bh = tf.Variable(np.zeros((1, dim), dtype=np.float32))
    stddev = 1.0 / math.sqrt(dim)
    Wx = tf.Variable(tf.truncated_normal([dim, dim], -0.1, stddev=stddev))
    Wy = tf.Variable(tf.truncated_normal([dim, dim], -0.1, stddev=stddev))
    Wh = tf.Variable(tf.truncated_normal([dim, dim], -0.1, stddev=stddev))
    def rnn(ix, ih):
        # h = tf.nn.sigmoid(tf.nn.xw_plus_b(ix, Wx, bx)+ tf.nn.xw_plus_b(h, Wh, bh))
        h = tf.nn.tanh( tf.matmul(ix, Wx) + bx + tf.matmul(ih, Wh) + bh)
        # h = tf.nn.relu( tf.matmul(ix, Wx) + bx + tf.matmul(ih, Wh) + bh)
        # y = tf.nn.xw_plus_b(h, Wy, by)
        # y = tf.nn.sigmoid( tf.matmul(h, Wy) + by)
        y = tf.nn.tanh(tf.matmul(h, Wy) + by)
        # y = tf.matmul(h, Wy) + by
        return y, h
    # Gather training rollings
    train_data = list()     # placeholder of (num_unrolling x batchsize, dim)
    for _ in range(num_unrollings +1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size, dim]))
    train_x = train_data[:-1]
    train_y = train_data[1:]
    # Unrolling
    outputs = list()
    h = np.zeros((1, dim), dtype=np.float32)
    for i in train_x:
        y, h = rnn(i, h)
        outputs.append(y)
    train_prediction = tf.concat(0, outputs)
    loss = tf.reduce_mean( 
        tf.square(
            train_prediction -  tf.concat(0, train_y)))
    # loss += tf.reduce_sum(tf.square(Wx)) + tf.reduce_sum(tf.square(Wh)) + tf.reduce_sum(tf.square(Wy))
    # Trian
    # ...
    # global_step = tf.Variable(0)
    # learning_rate = tf.train.exponential_decay(
    #   0.5, global_step, 250, 0.1, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # gradients, v = zip(*optimizer.compute_gradients(loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    # optimizer = optimizer.apply_gradients(
    #   zip(gradients, v), global_step=global_step)
    learning_rate = 0.1
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.999, 0.01).minimize(loss)




num_steps = 1000
summary_frequency = 10
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = batch
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    # _, l, predictions, lr = session.run(
    #   [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f' % (step, mean_loss))
        # 'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      # if step % (summary_frequency * 10) == 0:
      #   # Generate some samples.
      #   print('=' * 80)
      #   p = session.run(train_prediction, feed_dict=feed_dict)
      #   print(p)
      #   print('=' * 80)
  pre = session.run(train_prediction, feed_dict=feed_dict)
  session.run([Wx, Wh, Wy, bx, bh, by])

b = np.zeros((10,2))
b[:,0] = x[1:11, 0]
b[:,1] = pre[:10, 0]
b

# np.mean(np.square(p[:, 0] - x[:, 0]))
