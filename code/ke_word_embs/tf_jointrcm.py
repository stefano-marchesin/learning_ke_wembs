import math
import tensorflow as tf


class JointRCM(object):
    """build the graph for Word2Vec model"""
    def __init__(self, _word_vocab_size, _syns, _options):
        self.word_vocab_size = _word_vocab_size
        self.syns = _syns
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
            self.syn_labels = tf.reshape(tf.convert_to_tensor(self.syns[:,0], dtype=tf.int64), shape=[self.syns.shape[0], 1])

        """EMBEDDING LOOKUPS"""
        with tf.name_scope('lookups'):
            # word embedding lookups
            self.words = tf.nn.embedding_lookup(self.word_embs, self.inputs)
            # synonyms embedding lookups - each synonym is considered as the context of the given word (i.e. label)
            self.syn_context = tf.nn.embedding_lookup(self.word_embs, self.syns[:, 1])

        """FORWARD PASS"""
        with tf.name_scope('context_pass'):
            # take the mean of context words embeddings to generate the context embedding
            self.context = tf.reduce_mean(self.words, 1)

        """LOSS OPERATION"""
        with tf.name_scope('loss_ops'):
            # compute the average Noise Contrastive Estimation (NCE) loss for the given batch (cbow objective)
            self.cbow_loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.labels, self.context, opts.neg_samples, self.word_vocab_size))
            # compute the averate Noise Contrastive Estimation (NCE) loss for the synonyms provided (rmc objective)
            self.rcm_loss = opts.prior * tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.syn_labels, self.syn_context, opts.neg_samples, self.word_vocab_size))  # @smarchesin TODO: divide by self.word_vocab_size? see Yu & Dredze 2014

        """OPTIMIZATION OPERATION"""
        with tf.name_scope('opt_ops'): 
            # optimize cbow 
            # cbow_optimizer = tf.train.GradientDescentOptimizer(opts.cbow_lr)
            cbow_optimizer = tf.train.AdagradOptimizer(opts.cbow_lr)
            self.cbow_train_op = cbow_optimizer.minimize(self.cbow_loss)
            # optimize rcm
            # rcm_optimizer = tf.train.GradientDescentOptimizer(opts.rcm_lr)
            rcm_optimizer = tf.train.AdagradOptimizer(opts.rcm_lr)
            self.rcm_train_op = rcm_optimizer.minimize(self.rcm_loss)