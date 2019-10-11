import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import subprocess
import collections 
import sklearn.model_selection
import pytrec_eval
import json
import numpy as np 
import tensorflow as tf

from tqdm import tqdm
from cvangysel import trec_utils

import tf_utils


flags = argparse.ArgumentParser()

flags.add_argument("--run_path", default="", type=str, help="Lexical baseline run.")
flags.add_argument("--semantic_model", default="", type=str, help="Path to stored semantic model.")
flags.add_argument("--qrels_fname", default="qrels_short", type=str, help="Qrels filename.")
flags.add_argument("--query_fname", default="topics_orig", type=str, help="Query filename.")
flags.add_argument("--qfield", default="desc", type=str, help="Query field to consider for retrieval.")
flags.add_argument("--ref_measure", default="P_10", type=str, help="Reference measure to consider for optimization.")
flags.add_argument("--num_folds", default=2, type=int, help="Number of folds to consider for cross validation.")
flags.add_argument("--fixed_gamma", default=0.55, type=float, help="Perform re-ranking using the fixed gamma value instead of 2-fold cross validation.")
flags.add_argument("--sweep", default=0.05, type=float, help="Sweep value for optimized weights.")
flags.add_argument("--seed", default=42, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--corpus_name", default='OHSUMED_ALLTYPES', type=str, help="Corpus to consider.")
flags.add_argument("--model_name", default='lm_ccbow_alpha_1', type=str, help="Model name.")
flags.add_argument("--normalizer", default="minmax", type=str, help="Selected normalizer - possible normalizers: 'standardize', 'minmax', 'none'.")

FLAGS = flags.parse_args()


class StandardizationNormalizer(object):
	# apply standard deviation normalization
	def __init__(self, scores):
		self.mean = np.mean(scores)
		self.std = np.std(scores)

	def __call__(self, score):
		return (score - self.mean) / self.std


class MinMaxNormalizer(object):
	# apply minmax normalization
	def __init__(self, scores):
		self.min = np.min(scores)
		self.max = np.max(scores)

	def __call__(self, score):
		return (score - self.min) / (self.max - self.min)


class IdentityNormalizer(object):
	# apply identify normalization
	def __init__(self, scores):
		pass

	def __call__(self, score):
		return score


SCORE_NORMALIZERS = {
	'standardize': StandardizationNormalizer,
	'minmax': MinMaxNormalizer,
	'none': IdentityNormalizer
}


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))

	# set folders
	corpus_folder = 'corpus/' + FLAGS.corpus_name + '/' + FLAGS.corpus_name
	data_folder = 'corpus/' + FLAGS.corpus_name + '/data'
	query_folder = 'corpus/' + FLAGS.corpus_name + '/queries'
	qrels_folder = 'corpus/' + FLAGS.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + FLAGS.corpus_name + '/rankings/' + FLAGS.model_name

	# create folders 
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# parse and store qrels
	if FLAGS.qrels_fname:
		with open(qrels_folder + '/' + FLAGS.qrels_fname + '.txt', 'r') as qrelf:
			qrels = pytrec_eval.parse_qrel(qrelf)
		# initialize evaluator over qrels
		evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'P'})  # evaluate on Precision
	else:
		print("please provide qrels filename")
		return False

	"""
	LEXICAL PREPROCESSING
	"""

	# parse input run
	print('parse input run')
	with open(FLAGS.run_path, 'r') as runf:
		run = pytrec_eval.parse_run(runf)

	"""
	SEMANTIC PREPROCESSING
	"""

	# load required data
	print('load processed data required to perform re-ranking over lexical model w/ semantic model')
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
   
	# store docnos and docs as separate lists
	docnos = list(corpus.keys())
	docs = list(corpus.values())
	del corpus  # free memory space

	# load semantic model
	print('load semantic model')
	with tf.Session() as sess:
		# restore model and get required tensors
		saver = tf.train.import_meta_graph(FLAGS.semantic_model + '.ckpt.meta')
		saver.restore(sess, FLAGS.semantic_model + '.ckpt')
		word_embs = sess.run(tf.get_default_graph().get_tensor_by_name('embeddings/word_embs:0'))
	# compute doc embeddings
	doc_embs, filt_ids = tf_utils.compute_doc_embs(docs, word_dict, word_embs, idfs)

	"""
	COMPUTE RE-RANKING
	"""

	# set random seed
	np.random.seed(FLAGS.seed)
	# load queries
	q = tf_utils.read_ohsu_queries(query_folder + '/' + FLAGS.query_fname)
	# get query ids
	qids = list(q.keys())
	# shuffle query ids
	np.random.shuffle(qids)

	if FLAGS.fixed_gamma:  
		# perform re-ranking based on a fixed value of gamma
		print('perform re-ranking w/ gamma=%.2f' % (FLAGS.fixed_gamma))
		# initialize combined (output) run
		crun = trec_utils.OnlineTRECRun(FLAGS.model_name + '_gamma_' + str(FLAGS.fixed_gamma))
		# combine rankings using fixed gamma
		comb_run = tf_utils.compute_combined_run(run, FLAGS.qfield, q, docnos, doc_embs, word_dict, word_embs, SCORE_NORMALIZERS[FLAGS.normalizer], FLAGS.fixed_gamma)
		# store test ranking in combined run
		for qid, doc_ids_and_scores in comb_run.items():
			crun.add_ranking(qid, [(score, docno) for docno, score in doc_ids_and_scores.items()])
		# close and store run 
		crun.close_and_write(out_path=rankings_folder + '/' + FLAGS.model_name + '_gamma_' + str(FLAGS.fixed_gamma) + '.txt', overwrite=True)
		print('combined run stored in {}'.format(rankings_folder))
		# evalaute combined run
		print('evaluate run combined w/ gamma=%.2f' % (FLAGS.fixed_gamma))
		tf_utils.evaluate(['map', 'P_10', 'ndcg'], rankings_folder, FLAGS.model_name + '_gamma_' + str(FLAGS.fixed_gamma), qrels_folder, FLAGS.qrels_fname)
	else:
		# learn optimal weight to combine runs
		print("learn optimal weight to combine runs with sweep: {}".format(FLAGS.sweep))
		# set variable to store scores and weights
		scores_and_weights = []
		
		# initialize kfold with FLAGS.num_folds
		kfold = sklearn.model_selection.KFold(n_splits=FLAGS.num_folds)
		for fold, (train_qids, test_qids) in enumerate(kfold.split(qids)):
			print('fold n. {}'.format(fold))
			# restrict queries to train_qids and test_qids
			qtrain = {qids[ix]: q[qids[ix]] for ix in train_qids}
			qtest = {qids[ix]: q[qids[ix]] for ix in test_qids}
			# obtain best combination on training queries
			train_score, best_train_weight = max(tf_utils.perform_reranking(run, FLAGS.qfield, qtrain, docnos, doc_embs, word_dict, word_embs, FLAGS.sweep, SCORE_NORMALIZERS[FLAGS.normalizer], FLAGS.ref_measure, evaluator))
			print('fold %d: best_train_weight=%.2f, %s =%.4f' % (fold, best_train_weight, FLAGS.ref_measure, train_score))
			# compute combined run with best combination on test queries
			test_crun = tf_utils.compute_combined_run(run, FLAGS.qfield, qtest, docnos, doc_embs, word_dict, word_embs, SCORE_NORMALIZERS[FLAGS.normalizer], best_train_weight)
			# evaluate test run
			test_res = evaluator.evaluate(test_crun)
			# compute aggregated measure score for test queries
			test_score = pytrec_eval.compute_aggregated_measure(FLAGS.ref_measure, [qscore[FLAGS.ref_measure] for qscore in test_res.values()])
			# store averaged scores w/ best weights
			scores_and_weights.append((np.mean([train_score, test_score]), best_train_weight))

		# get (best) weight that produces the highest averaged score
		best_score, best_weight = max(scores_and_weights)
		print('found best weight=%.2f' % (best_weight))
		# initialize combined (output) run
		crun = trec_utils.OnlineTRECRun(FLAGS.model_name + '_best_weight_' + str(FLAGS.best_weight))
		# compute combined run based on test weight
		comb_run = tf_utils.compute_combined_run(run, FLAGS.qfield, q, docnos, doc_embs, word_dict, word_embs, SCORE_NORMALIZERS[FLAGS.normalizer], best_weight) 
		# store ranking in crun
		for qid, doc_ids_and_scores in comb_run.items():
			crun.add_ranking(qid, [(score, doc_id) for doc_id, score in doc_ids_and_scores.items()])
		# close and store run
		crun.close_and_write(out_path=rankings_folder + '/' + FLAGS.model_name + '_best_weight_' + str(FLAGS.best_weight) + '.txt', overwrite=True)
		print('combined run stored in {}'.format(rankings_folder))
		# evalaute combined run
		print('evaluate run combined w/ {}-fold cross validation and best weight={}'.format(FLAGS.num_folds, FLAGS.best_weight))
		tf_utils.evaluate(['map', 'P_10', 'ndcg'], rankings_folder, FLAGS.model_name + '_best_weight_' + str(FLAGS.best_weight), qrels_folder, FLAGS.qrels_fname)


if __name__ == "__main__":
	main()  


