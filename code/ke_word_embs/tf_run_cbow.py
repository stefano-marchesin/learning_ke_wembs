import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import numpy as np
import tensorflow as tf

from tqdm import tqdm 

import tf_utils
import tf_globals

from tf_cbow import CBOW


flags = tf.app.flags

flags.DEFINE_integer("word_embs_size", 300, "The word embedding dimension size.")
flags.DEFINE_integer("epochs", 1,
	"Number of epochs to train. Each epoch processes the training data once completely.")
flags.DEFINE_integer("negative_samples", 10,
					 "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 128,
					 "Number of training examples processed per step "
					 "(size of a minibatch).")
flags.DEFINE_integer("min_cut_freq", 5, 
					 "Minimum word frequency allowed.")
flags.DEFINE_integer("context_window", 5,
					 "The number of words to predict to the left or right "
					 "of the target word.")
flags.DEFINE_float("learning_rate", 0.025, "Learning rate.")
flags.DEFINE_integer("seed", 42,
					 "Answer to ultimate question of life, the universe and everything.")
flags.DEFINE_string("query_field", "desc", "Query target field.")
flags.DEFINE_string("corpus_name", "OHSUMED_ALLTYPES", "Target corpus name.")
flags.DEFINE_string("query_fname", "topics_orig", "Query file name.")
flags.DEFINE_string("qrels_fname", "qrels_short", "Qrels file name.")
flags.DEFINE_string("inf_qrels_fname", "qrels-sampleval-2014", "Inferred qrels file name.")
flags.DEFINE_string("reference_measure", "P_10", "Reference measure to be used for model optimization.")
flags.DEFINE_string("model_name", "tf_cbow_adag_lr0025_ohsu", "Model name.")
FLAGS = flags.FLAGS


class Options(object):
	"""options used by the Neural Vector Space Model (NVSM)"""

	def __init__(self):
		# word embeddings dimension
		self.word_size = FLAGS.word_embs_size
		# number of negative samples per example
		self.neg_samples = FLAGS.negative_samples
		# epochs to train
		self.epochs = FLAGS.epochs
		# batch size
		self.batch_size = FLAGS.batch_size
		# dict size
		self.min_cut_freq = FLAGS.min_cut_freq
		# context window size
		self.context_window = FLAGS.context_window
		# learning rate
		self.lr = FLAGS.learning_rate
		# seed
		self.seed = FLAGS.seed
		# query field
		self.field = FLAGS.query_field
		# corpus name
		self.corpus_name = FLAGS.corpus_name
		# query file name
		self.query_fname = FLAGS.query_fname
		# qrels file name
		self.qrels_fname = FLAGS.qrels_fname
		# inferred qrels file name
		self.inf_qrels_fname = FLAGS.inf_qrels_fname
		# reference measure
		self.ref_measure = FLAGS.reference_measure
		# model name
		self.model_name = FLAGS.model_name


def main(_):
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set folders
	corpus_folder = 'corpus/' + opts.corpus_name + '/' + opts.corpus_name
	index_folder = 'corpus/' + opts.corpus_name + '/index'
	model_folder = 'corpus/' + opts.corpus_name + '/models/' + opts.model_name
	data_folder = 'corpus/' + opts.corpus_name + '/data'
	query_folder = 'corpus/' + opts.corpus_name + '/queries'
	qrels_folder = 'corpus/' + opts.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + opts.corpus_name + '/rankings/' + opts.model_name

	# create folders 
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)
	if not os.path.exists(index_folder):
		os.makedirs(index_folder)
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# load queries
	q = tf_utils.read_ohsu_queries(query_folder + '/' + opts.query_fname)

	"""
	PRE PROCESSING
	"""

	# pre process distributional data
	if not os.path.exists(data_folder + '/words.json'):
		# compute required data
		words = tf_utils.process_corpus(corpus_folder, data_folder)
		# build dataset to train CBOW + RMC model
		data, cfs, word_dict, reverse_word_dict = tf_utils.build_dataset(words, opts.min_cut_freq, data_folder)
		del words  # free memory from unnecessary data
		print('Most common words (+ UNK)', count[:10])
		print('Total number of words (+ UNK) within {}: {}'.format(opts.corpus_name, len(data)))
		print('Number of unique words (+ UNK) for {}: {}'.format(opts.corpus_name, len(count)))
	else:
		# load required data
		print('load processed data required to train CBOW model')
		with open(data_folder + '/data.json', 'r') as df:
			data = json.load(df)
		with open(data_folder + '/docs.json', 'r') as cf:
			corpus = json.load(cf)
		with open(data_folder + '/idfs.json', 'r') as wf:
			idfs = json.load(wf)
		with open(data_folder + '/cfs.json', 'r') as cff:
			cfs = json.load(cff)
		with open(data_folder + '/word_dict.json', 'r') as wdf:
			word_dict = json.load(wdf)
		# compute reverse word dictionary 
		reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))

	# load required data to perform retrieval
	print('load required data to perform retrieval')
	with open(data_folder + '/docs.json', 'r') as cf:
		corpus = json.load(cf)
	with open(data_folder + '/idfs.json', 'r') as wf:
		idfs = json.load(wf)
	# get docs and docnos from corpus
	docnos = list(corpus.keys())
	docs = list(corpus.values())
	del corpus  # free memory space

	"""
	NETWORK TRAINING
	"""

	# begin training 
	with tf.Graph().as_default(), tf.Session() as sess:
		# set graph-level random seed
		tf.set_random_seed(opts.seed)
		# start data index
		tf_globals.initialize()
		# setup the model
		model = CBOW(len(word_dict), opts)
		# create model saving operation - keeps as many saved models as number of epochs
		saver = tf.train.Saver(max_to_keep=opts.epochs)
		# initialize the variables using global_variables_initializer()
		sess.run(tf.global_variables_initializer())

		print('start training')
		print('number of batches per epoch: {}'.format(len(data) // opts.batch_size))
		best_score_per_epoch = []
		for epoch in range(opts.epochs):
			# train CBOW
			print('training epoch {}'.format(epoch + 1))
			# loop over (len(data) // opts.batch_size) batches
			for i in tqdm(range(len(data) // opts.batch_size)): 
				batch_inputs, batch_labels = tf_utils.generate_batch(data, opts.batch_size, opts.context_window)
				feed_dict = {model.inputs: batch_inputs, model.labels: batch_labels}
				# run train_op
				sess.run(model.train_op, feed_dict=feed_dict)
			# store trained CBOW
			print('storing model at epoch {}'.format(epoch + 1))
			model_checkpoint_path = os.path.join(os.getcwd(), model_folder, opts.model_name + str(epoch + 1) + '.ckpt')
			save_path = saver.save(sess, model_checkpoint_path)
			print("model saved in file: {}".format(save_path))
			
			"""
			DOCUMENT RETRIEVAL 
			"""
			
			# get embs after training epoch
			word_embs = sess.run(model.word_embs)
			# evaluate CBOW for IR tasks
			print('evaluating at epoch {}'.format(epoch + 1))
			# compute doc embeddings and return list of filtered doc ids
			doc_embs, filt_ids = tf_utils.compute_doc_embs(docs, word_dict, word_embs, idfs)
			# set query embs and ids
			q_embs = []
			q_ids = []
			# loop over queries and generate rankings
			for qid, qtext in q.items():
				# prepare queries for semantic matching
				q_proj = tf_utils.prepare_query(qtext[opts.field], word_dict, word_embs)
				if q_proj is None:
					print('query {} does not contain known terms'.format(qid))
				else:
					q_embs.append(q_proj)
					q_ids.append(qid)
			q_embs = np.array(q_embs)
			# perform search and evaluate model effectiveness
			tf_utils.semantic_search(docnos, doc_embs, q_ids, q_embs, rankings_folder, opts.model_name + '_' + str(epoch + 1), filt_ids)
			scores = tf_utils.evaluate(['Rprec', 'P_5', 'P_10', 'P_20', 'ndcg',  'map'], rankings_folder, opts.model_name + '_' + str(epoch + 1), qrels_folder, opts.qrels_fname)
			best_score_per_epoch.append(scores[opts.ref_measure])
	print('best model (in terms of {}) found at epoch: {}'.format(opts.ref_measure, np.argsort(best_score_per_epoch)[-1] + 1))


if __name__ == "__main__":
	tf.app.run() 