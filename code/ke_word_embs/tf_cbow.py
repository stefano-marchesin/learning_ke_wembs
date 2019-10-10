import math
import tensorflow as tf


class CBOW(object):
    """build the graph for Word2Vec model"""
    def __init__(self, _word_vocab_size, _options):
        self.word_vocab_size = _word_vocab_size
        self.options = _options

        """PARAMETER INITIALIZATION"""
        opts = self.options

        with tf.name_scope('embeddings'):
            self.word_embs = tf.Variable(tf.random_uniform([self.word_vocab_size, opts.word_size], -1.0, 1.0), name='word_embs')

        with tf.name_scope('nce_weights'):
            self.nce_weights = tf.Variable(tf.truncated_normal([self.word_vocab_size, opts.word_size], stddev=1.0 / math.sqrt(opts.word_size)), name='nce_weights')

        with tf.name_scope('nce_biases'):
            self.nce_biases = tf.Variable(tf.zeros([self.word_vocab_size]), name='nce_biases')

        """PLACEHOLDERS"""
        with tf.name_scope('placeholders'):
            self.inputs = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.context_window * 2])
            self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size, 1])

        """EMBEDDING LOOKUPS"""
        with tf.name_scope('lookups'):
            # embedding lookups
            self.words = tf.nn.embedding_lookup(self.word_embs, self.inputs)

        """FORWARD PASS"""
        with tf.name_scope('context_pass'):
            # take the mean of context words embeddings to generate the context embedding
            self.context = tf.reduce_mean(self.words, 1)

        """LOSS OPERATION"""
        with tf.name_scope('loss_ops'):
            # compute the average Noise Contrastive Estimation (NCE) loss for the given batch
            self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.labels, self.context, opts.neg_samples, self.word_vocab_size))
            # compute negative sampling loss for the given batch
                # self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.nce_weights, self.nce_biases, self.labels, self.context, opts.neg_samples, self.word_vocab_size))

        """OPTIMIZATION OPERATION"""
        with tf.name_scope('opt_ops'):  
            # optimizer = tf.train.GradientDescentOptimizer(opts.lr)
            optimizer = tf.train.AdagradOptimizer(opts.lr)
            self.train_op = optimizer.minimize(self.loss)