import math
import tensorflow as tf
import numpy as np

from tf_repeat import gather_repeat


class CCBOW(object):
	"""build the graph for the constrained Word2Vec CBOW model"""
	def __init__(self, _word_vocab_size, _synsets, _weights, _options):
		self.word_vocab_size = _word_vocab_size
		self.synsets = tf.ragged.constant(_synsets)
		self.weights = tf.ragged.constant(_weights)
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

		"""SYNONYMS GENERATION"""
		with tf.name_scope('syns_generation'):
			# gather synonyms for labels (i.e. target words)
			self.syns = tf.gather(self.synsets, tf.squeeze(self.labels))
			# reshape synonyms to be compliant with tf.nn.nce_loss input
			self.syn_labels = tf.reshape(self.syns.values, shape=[-1, 1])
			# store the length of each synset associated to self.labels
			self.syn_lens = self.syns.row_lengths()
			# pre compute the normalized weights for each word in vocabulary given the following formula: W(ws|wt) = cf(ws) / sum([cf(w) for w in syns(wt)])
			self.weights = tf.to_float(self.weights / tf.reshape(tf.reduce_sum(self.weights, axis=1), shape=[-1, 1]))
			
		"""EMBEDDING LOOKUPS"""
		with tf.name_scope('lookups'):
			# word embedding lookups
			self.words = tf.nn.embedding_lookup(self.word_embs, self.inputs)

		"""FORWARD PASS"""
		with tf.name_scope('context_pass'):
			# take the mean of context words embeddings to generate the context embedding
			self.contexts = tf.reduce_mean(self.words, 1)
			
		"""LOSS OPERATION"""
		with tf.name_scope('loss_ops'):
			# compute the Noise Contrastive Estimation (NCE) loss for the given batch (cbow objective)
			self.cbow_loss = tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.labels, self.contexts, opts.neg_samples, self.word_vocab_size)
			# condition rule to decide whether to compute synonymy-enhanced loss - when batch does not contain any synset switch to tf.constant(0.0)
			self.condition = tf.count_nonzero(self.syn_lens)
			# perform lazy tf.cond and compute loss (syns objective)
			true_fn = lambda: self.compute_syn_loss()
			self.syn_loss = tf.cond(self.condition > 0, true_fn=true_fn, false_fn=lambda: tf.constant(0.0, dtype=tf.float32))
			# combine losses
			self.loss = tf.reduce_sum(self.cbow_loss) - opts.alpha * self.syn_loss
			self.loss /= opts.batch_size

		"""OPTIMIZATION OPERATION"""
		with tf.name_scope('opt_ops'): 
			# optimize constained cbow 
			optimizer = tf.train.AdagradOptimizer(opts.lr)
			self.train_op = optimizer.minimize(self.loss)

	def compute_syn_loss(self):
		"""compute the loss derived from the synonymy-enhanced component"""
		self.syn_contexts = gather_repeat(self.contexts, self.syn_lens)
		self.syns_loss = tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.syn_labels , self.syn_contexts, self.options.neg_samples, self.word_vocab_size)
		self.syn_weights = tf.gather(self.weights, tf.squeeze(self.labels)).values
		return tf.reduce_sum(tf.multiply(self.syn_weights, tf.square(gather_repeat(self.cbow_loss, self.syn_lens) - self.syns_loss)))
