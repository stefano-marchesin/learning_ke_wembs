import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import subprocess
import collections 
import sklearn.model_selection
import pytrec_eval
import json
import math
import re
import sys
import numpy as np 
import tensorflow as tf

from copy import deepcopy
from tqdm import tqdm
from whoosh.analysis import SimpleAnalyzer
from gensim.corpora import Dictionary
from cvangysel import trec_utils

import tf_utils
import umls


flags = argparse.ArgumentParser()

flags.add_argument("--run_path", default="", type=str, help="Lexical baseline run.")
flags.add_argument("--semantic_model", default="", type=str, help="Path to stored semantic model.")
flags.add_argument("--retrofit", default=True, type=bool, help="Whether to retrofit word vectors using lexicon information.")
flags.add_argument("--iterations", default=10, type=int, help="Number of iteration to retrofit word vectors using lexicon information.")
flags.add_argument("--beta", default=1.0, type=float, help="Regularization parameter to retrofit word vectors through semantic lexicons.")
flags.add_argument("--syn_weights", default=True, type=bool, help="Applies term weighting to word synonyms based on terms relative collection frequencies.")
flags.add_argument("--reranking", default=True, type=bool, help="Whether to evaluate model on a ranking or re-ranking task.")
flags.add_argument("--qrels_fname", default="qrels_short", type=str, help="Qrels filename.")
flags.add_argument("--query_fname", default="topics_orig", type=str, help="Query filename.")
flags.add_argument("--qfield", default="desc", type=str, help="Query field to consider for retrieval.")
flags.add_argument("--ref_measure", default="P_10", type=str, help="Reference measure to consider for optimization.")
flags.add_argument("--num_folds", default=2, type=int, help="Number of folds to consider for cross validation.")
flags.add_argument("--fixed_gamma", default=0.55, type=float, help="Perform re-ranking using the fixed gamma value instead of 2-fold cross validation.")
flags.add_argument("--sweep", default=0.05, type=float, help="Sweep value for optimized weights.")
flags.add_argument("--seed", default=42, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--corpus_name", default='OHSUMED_ALLTYPES', type=str, help="Corpus to consider.")
flags.add_argument("--model_name", default='lm_crf_beta_1', type=str, help="Model name.")
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


def norm_word(word):
	return word.lower()


def norm_embs(word_embs):
	"""read all the word vectors and normalize them"""
	norm_embs = []
	for emb in word_embs:
		# normalize weight vector
		norm_embs.append(emb / math.sqrt((emb**2).sum() + 1e-6))
	return np.array(norm_embs)


def retrofit(word_embs, syns, reverse_dict, num_iters, alpha=1.0, beta=1.0, cfs=None):
	"""retrofit word vectors to a lexicon"""
	new_embs = deepcopy(word_embs)
	for it in range(num_iters):
		# loop through every node also in ontology (else just use data estimate)
		for word, synset in tqdm(syns.items()):
			num_syns = len(synset)
			# no synonyms, pass - use data estimate
			if num_syns == 0:
				continue
			# the weight of the data estimate if the number of neighbours
			new_emb = (num_syns * alpha) * word_embs[word]
			# loop over synonyms and add to new vector
			weights = []
			for ix, syn in enumerate(synset):
				# the weight of the current synonym
				if cfs:
					syn_weight = beta * (cfs[reverse_dict[syn]] / np.sum([cfs[reverse_dict[s]] for s in synset]))
				else:
					syn_weight = 1 / num_syns  # synonyms weights are set to degree(ix)^-1 - i.e. number of synonyms that ix has
				weights.append(syn_weight)
				new_emb += new_embs[syn] * weights[ix]
			new_embs[word] = new_emb / (np.sum(weights) + (num_syns * alpha))
	return new_embs


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# set folders
	corpus_folder = 'corpus/' + FLAGS.corpus_name + '/' + FLAGS.corpus_name
	index_folder = 'corpus/' + FLAGS.corpus_name + '/index'
		# model_folder = 'corpus/' + FLAGS.corpus_name + '/models/' + FLAGS.model_name
	data_folder = 'corpus/' + FLAGS.corpus_name + '/data'
	query_folder = 'corpus/' + FLAGS.corpus_name + '/queries'
	qrels_folder = 'corpus/' + FLAGS.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + FLAGS.corpus_name + '/rankings/' + FLAGS.model_name 

	# create folders 
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
		# if not os.path.exists(model_folder):
			# os.makedirs(model_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# set random seed - enable reproducibility
	np.random.seed(FLAGS.seed)
	# establish connection with UMLS db
	umls_lookup = umls.UMLSLookup()

	# load required data
	print('load processed data required to retrofit word vectors and perform retrieval tasks')
	with open(data_folder + '/docs.json', 'r') as df:
		corpus = json.load(df)
	with open(data_folder + '/idfs.json', 'r') as wf:
		idfs = json.load(wf)
	with open(data_folder + '/cfs.json', 'r') as cff:
		cfs = json.load(cff)
	with open(data_folder + '/word_dict.json', 'r') as wdf:
		word_dict = json.load(wdf)
	# compute reverse word dict
	reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))
	# store docnos and docs as separate lists
	docnos = list(corpus.keys())
	docs = list(corpus.values())
	del corpus  # free memory space
	
	# pre process relational data
	if not os.path.exists(data_folder + '/term2cui.json'):
		# map terms to cuis using QuickUMLS
		term2cui = tf_utils.get_term2cui(word_dict, data_folder, threshold=FLAGS.threshold, stypes_fname=FLAGS.stypes_fname)
	else: 
		# laod (term, cui) pairs 
		print('load (term, cui) pairs')
		with open(data_folder + '/term2cui.json', 'r') as tcf:
			term2cui = json.load(tcf)

	
	"""
	SEMANTIC PROCESSING
	"""	

	# load semantic model
	print('load semantic model')
	with tf.Session() as sess:
		# restore model and get required tensors
		saver = tf.train.import_meta_graph(FLAGS.semantic_model + '.ckpt.meta')
		saver.restore(sess, FLAGS.semantic_model + '.ckpt')
		word_embs = sess.run(tf.get_default_graph().get_tensor_by_name('embeddings/word_embs:0'))
	
	"""
	RETROFITTING
	"""

	if FLAGS.retrofit:
		# get synonyms for each word within vocabulary
		print('get synonyms')
		syns = tf_utils.get_syns(term2cui, word_dict, umls_lookup)
		if FLAGS.syn_weights:
			# convert collection frequencies from list to dict
			cfs = dict(cfs)
		else:
			cfs = None
		# retrofit word vectors 
		print('retrofit word vectors for {} iterations'.format(FLAGS.iterations))
		word_embs = retrofit(word_embs, syns, reverse_word_dict, FLAGS.iterations, alpha=1.0, beta=FLAGS.beta, cfs=cfs)

	# compute doc embeddings
	print('compute document vectors w/ retrofitted word vectors')
	doc_embs, filt_ids = tf_utils.compute_doc_embs(docs, word_dict, word_embs, idfs)

	if not FLAGS.reranking:
		
		"""
		RETRIEVAL
		"""
		print('perform retrieval over the entire collection')
		# load queries
		q = tf_utils.read_ohsu_queries(query_folder + '/' + FLAGS.query_fname)
		# set query embs and ids
		q_embs = []
		q_ids = []
		# loop over queries and generate rankings
		for qid, qtext in q.items():
			# prepare queries for semantic matching
			q_proj = tf_utils.prepare_query(qtext[FLAGS.qfield], word_dict, word_embs)
			if q_proj is None:
				print('query {} does not contain known terms'.format(qid))
			else:
				q_embs.append(q_proj)
				q_ids.append(qid)
		q_embs = np.array(q_embs)
		# perform search and evaluate model effectiveness
		tf_utils.semantic_search(docnos, doc_embs, q_ids, q_embs, rankings_folder, FLAGS.model_name)
		scores = tf_utils.evaluate(['Rprec', 'P_5', 'P_10', 'P_20', 'ndcg',  'map'], rankings_folder, FLAGS.model_name, qrels_folder, FLAGS.qrels_fname)

	else: 
		
		"""
		RE-RANKING
		"""
		print('perform re-ranking over top 1000 documents from a baseline run')
		# parse and store qrels
		with open(qrels_folder + '/' + FLAGS.qrels_fname + '.txt', 'r') as qrelf:
			qrels = pytrec_eval.parse_qrel(qrelf)
		# initialize evaluator over qrels
		evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'P'})  # evaluate on Precision
	   
		# parse input run
		print('parse input run')
		with open(FLAGS.run_path, 'r') as runf:
			run = pytrec_eval.parse_run(runf)
		
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
	