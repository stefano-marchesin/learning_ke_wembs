import os
import collections
import math
import glob
import json
import subprocess
import numpy as np
import pytrec_eval
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from textwrap import wrap
from whoosh.analysis import SimpleAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

import tf_globals

from QuickUMLS.quickumls.client import get_quickumls_client
from quickumls_conn import QuickUMLS


def load_stypes(stypes_path):
    """read semantic types as list"""
    with open(stypes_path, 'r') as stf:
        stypes = [stype.split('|')[1] for stype in stf]
    return stypes


def read_trec_corpus(corpus_path):
    """convert trec style corpus into (valid) xml"""
    docs = glob.glob(corpus_path + '/**/*.txt', recursive=True)
    for doc in docs:
        with open(doc, 'r') as f:  # read doc
            xml = f.read()
        # convert into true xml
        xml = '<ROOT>' + xml + '</ROOT>'
        # fix bad-formed tokens
        xml = xml.replace('&', '&amp;')
        yield xml


def get_trec_doc(corpus_path):
    """generate doc from batch of TREC-style docs"""
    ohsu = read_trec_corpus(corpus_path)
    # loop over batches
    for batch in ohsu:
        # parse xml
        root = ETree.fromstring(batch)
        # loop through each doc in the batch
        for doc in tqdm(root):
            docno = ''
            body = ''
            # loop through each element (tag, value)
            for elem in doc:
                if elem.tag == 'DOCNO':
                    docno = elem.text.strip()
                else:
                    body = elem.text.strip()
            # return doc to index
            yield docno, body


def process_corpus(corpus_path, out_path):
    """process corpus: split docs into words and return tokenized corpus"""
    corpus = get_trec_doc(corpus_path)
    # set tokenizer
    tokenizer = SimpleAnalyzer()
    # tokenize corpus and store into words
    print("tokenizing corpus...")
    words = []
    dfreqs = {}
    docs = {}
    for docno, doc in corpus:
        # tokenize docs
        doc_tokens = [token.text for token in tokenizer(doc)]
        # assign tokens
        docs[docno] = doc_tokens
        words.extend(doc_tokens)
        # update doc frequencies
        for token in set(doc_tokens):
            if token in dfreqs:
                dfreqs[token] += 1
            else:
                dfreqs[token] = 1
    print("corpus tokenized!")
    print("computing IDF scores for words within corpus")
    idfs = {token: np.log(len(docs) / (1 + float(dfreq))) for token, dfreq in dfreqs.items()}
    print("store processed data")
    with open(out_path + '/words.json', 'w') as file_words:
        json.dump(words, file_words)
    with open(out_path + '/docs.json', 'w') as file_docs:
        json.dump(docs, file_docs)
    with open(out_path + '/idfs.json', 'w') as file_idfs:
        json.dump(idfs, file_idfs)
    return words


def build_dataset(words, min_cut_freq, out_path):
    """build data required to train Word2Vec models"""
    count_orig = [['UNK', -1]]
    # keep all words within vocabulary
    count_orig.extend(collections.Counter(words).most_common())
    count = [['UNK', -1]]
    for word, freq in count_orig:
        word_tuple = [word, freq]
        if word == 'UNK':
            count[0][1] = freq
            continue
        if freq >= min_cut_freq:
            count.append(word_tuple)
    word_dict = {}
    for word, _ in count:
        word_dict[word] = len(word_dict)
    data = []
    unk_count = 0
    for word in words:
        if word in word_dict:
            index = word_dict[word]
        else:
            index = 0  # word_dict['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))
    print("store encoded corpus and dictionary data")
    with open(out_path + '/data.json', 'w') as file_data:
        json.dump(data, file_data)
    with open(out_path + '/cfs.json', 'w') as file_cfs:
        json.dump(count, file_cfs)
    with open(out_path + '/word_dict.json', 'w') as file_dict:
        json.dump(word_dict, file_dict)
    # return computed data
    return data, count, word_dict, reverse_word_dict


def get_term2cui(word_dict, out_path, threshold=1.0, stypes_fname=None):
    """map candidate CUIs to each indexed word"""
    terms = ' '.join(list(word_dict.keys()))
    # split terms into substrings of length <= 999999 -- max length allowed by scipy parser
    subs = wrap(terms, width=999999, break_long_words=False, break_on_hyphens=False)
    if stypes_fname is not None:  # load user-specified UMLS semantic types
        print("user-specified UMLS semantic types for QuickUMLS enabled")
        stypes = ','.join(load_stypes(stypes_fname))
    else:  # keep default QuickUMLS semantic types
        stypes = None
    # initialize QuickUMLS server
    server = QuickUMLS(window=1, threshold=threshold, semtypes=stypes)
    server.launch_quickumls()
    # initialize concept matcher
    matcher = get_quickumls_client()
    term2cui = []
    # extract concepts
    for sub in subs:
        cuis = matcher.match(sub)
        # get position dict {pos: [term, ["__NULL__"]]} given sub
        pos2term = get_pos2term(sub) 
        # associate each term to its candidate CUIs
        term2cui += map_term2cui(pos2term, cuis)
    # close connection w/ QuickUMLS server
    server.close_quickumls()
    # store term2cui as a dictionary
    print("store (word, cui) pairs as a dictionary")
    term2cui = dict(term2cui)
    with open(out_path + '/term2cui.json', 'w') as file_t2c:
        json.dump(term2cui, file_t2c)
    return term2cui


def get_pos2term(text):
    """split text into terms and return {pos: [term, ["__NULL__"]]}"""
    pos2term = {}
    terms = text.split()  # split on whitespaces as text has been already pre processed
    # set text index
    index = text.index
    running_offset = 0
    # loop over terms
    for term in terms:
        term_offset = index(term, running_offset)
        term_len = len(term)
        # update running offset
        running_offset = term_offset + term_len
        pos2term[term_offset] = [term, "__NULL__"]  # note: "__NULL__" is for later use
    return pos2term


def map_term2cui(pos2term, cuis):
    """return list of (term, cui) pairs given term position and cuis"""
    for cui in cuis:
        # get positional info
        start = cui[0]['start']
        # check whether 'start' matches any pos2term key
        if start in pos2term:
            # update ["__NULL__"] w/ cui in first position (best candidate from QuickUMLS)
            pos2term[start][1] = cui[0]['cui']
    # return pos2term values only - i.e. (term, CUI) pairs
    return list(pos2term.values())


def get_syns(term2cui, word_dict, umls_lookup):
    """get synonymy relations from corpus and lexicon as a dictionary"""
    syns = {}
    analyzer = SimpleAnalyzer()
    for term, cui in term2cui.items():
        if cui != "__NULL__":
            synset = {word_dict[syn[0].lower()] for syn in umls_lookup.lookup_synonyms(cui, preferred=False) if len(list(analyzer(syn[0]))) == 1 and syn[0].lower() in word_dict and syn[0].lower() != term}
            if len(synset) > 0:
                syns[word_dict[term]] = list(synset)
            else: 
                syns[word_dict[term]] = []
        else:
            syns[word_dict[term]] = []
    return syns


def generate_batch(data, batch_size, context_window):
    """generate a tranining batch for the Word2Vec CBOW model"""
    context_size = 2 * context_window
    batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * context_window + 1  # [context_window target context_window]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[tf_globals.data_index])
        tf_globals.data_index = (tf_globals.data_index + 1) % len(data)
    for i in range(batch_size):
        # context words are just all the words within buffer except target
        batch[i, :] = [word for idx, word in enumerate(buffer) if idx != context_window]
        labels[i, 0] = buffer[context_window]
        buffer.append(data[tf_globals.data_index])
        tf_globals.data_index = (tf_globals.data_index + 1) % len(data)
    return batch, labels


def compute_doc_embs(docs, word_dict, word_embs, weights):
    """"compute doc embs as the weighted sum of their word embs"""
    doc_embs = []
    filtered_docs = []
    # loop over docs
    for idx, doc in tqdm(enumerate(docs)):
        if doc:  # doc is not empty
            doc_emb = np.array([word_embs[word_dict[term]] for term in doc if term in word_dict])
            doc_weights = np.array([weights[term] for term in doc if term in word_dict])
            doc_emb = np.sum(doc_emb * doc_weights[:, np.newaxis], axis=0)  # add an extra dimension to allow vector-scalar multiplication
        else:  # doc is empty
            filtered_docs.append(idx)
            continue
        doc_embs.append(doc_emb)
    print('number of document embeddings computed: {} \n number of skipped documents: {}'.format(len(doc_embs), len(filtered_docs)))
    # return np array of doc embs and list of filtered docs
    return np.array(doc_embs), filtered_docs


def read_ohsu_queries(query_path):
    """read query file and return a dict[id] = {title: <string>, desc: <string>}"""
    with open(query_path, 'r') as qf:
        q = qf.read()
    q = [query.split('\n') for query in q.split('\n\n') if query]
    # loop through each query and fill dict
    qdict = dict()
    for query in q:
        qid = query[1].split()[-1]
        qdict[qid] = dict()
        qdict[qid]['title'] = query[2].split('<title>')[1].strip()
        qdict[qid]['desc'] = query[4]
    return qdict


def tokenize_query(q):
    """lowerize and tokenize query"""
    analyzer = SimpleAnalyzer()
    return [token.text for token in analyzer(q)]


def project_query(q, word_dict, word_embs):
    """project list of terms into embedding space and sum them"""
    q_embs = np.array([word_embs[word_dict[term]] for term in q if term in word_dict])
    if q_embs.size == 0:
        return None
    else:
        return np.sum(q_embs, axis=0)


def prepare_query(q, word_dict, word_embs):
    """transform query into dense vector of size [1, embs]"""
    query_tokens = tokenize_query(q)
    query_proj = project_query(query_tokens, word_dict, word_embs)
    return query_proj


def semantic_search(docnos, doc_embs, query_ids, q_embs, ranking_folder, ranking_name, filtered_docs=None):
    """perform semantic search over docs given queries"""
    docnos = np.array(docnos)
    if filtered_docs:
        # remove filtered docs (i.e. docs with size = 0) from docnos
        docnos = np.delete(docnos, filtered_docs)
    # compute similarities
    print("compute similarities between docs and queries")
    similarities = cosine_similarity(doc_embs, q_embs)
    # open file to write results
    rf = open(ranking_folder + '/' + ranking_name + '.txt', 'w')
    # write results in ranking file
    for i in tqdm(range(similarities.shape[1])):
        rank = np.argsort(-similarities[:, i])[:1000]
        docs_rank = docnos[rank]
        qid = query_ids[i]
        # verify whether qid is an integer
        if qid.isdigit():  # cast to integer - this operation avoids storing topic ids as '059' instead of '59'
            qid = str(int(qid))  # convert to int and then back to str
        for j in range(len(docs_rank)):
            # write into .run file
            rf.write('%s %s %s %d %f %s\n' % (qid, 'Q0', docs_rank[j], j, similarities[rank[j]][i], ranking_name))
    rf.close()
    return True


def compute_combined_run(run, qfield, queries, docnos, doc_embs, word_dict, word_embs, normalizer, weight):
    """compute combined rank between input run and semantic model"""
    combined_run = {}
    # convert docnos into dicts {docno: idx} and {idx: docno}
    docno2idx = {docno: idx for idx, docno in enumerate(docnos)}
    idx2docno = {idx: docno for idx, docno in enumerate(docnos)}
    # loop over qids
    print('combine lexical/semantic models w/ weight: {}'.format(weight))
    for qid, qtext in tqdm(queries.items()):
        # query ranking
        qrankings = collections.defaultdict(list)
        # check whether run has ranking for current query
        if run[qid]:
            # get run ranking (lexical baseline)
            lex_ranking = run[qid] 
            # compute query embedding
            q_emb = prepare_query(qtext[qfield], word_dict, word_embs)
            if q_emb is not None:
                # get doc indexes from baseline (lexical) ranking
                doc_idxs = [docno2idx[docno] for docno in lex_ranking.keys()]
                # get hashmap from doc indexes
                idx2pos = {idx: pos for pos, idx in enumerate(doc_idxs)}
                # compute cosine similarity between query and doc embeddings
                cosine_scores = cosine_similarity(doc_embs[doc_idxs], np.array([q_emb]))
                # convert cosine_scores to dict {idx: score}
                sem_ranking = dict(enumerate(cosine_scores.flatten()))
                # compute ranking normalization
                lex_norm = normalizer(list(lex_ranking.values()))
                sem_norm = normalizer(list(sem_ranking.values()))
                # iterate over docs within run ranking 
                for docno, score in lex_ranking.items():
                    # append weighted (normalized) doc scores to qrankings
                    qrankings[docno].append(weight * lex_norm(score))  # lexical score
                    qrankings[docno].append((1-weight) * sem_norm(sem_ranking[idx2pos[docno2idx[docno]]]))  # semantic score
        # compute combined ranking for given query
        combined_run[qid] = {docno: np.sum(scores) for docno, scores in qrankings.items()}
    return combined_run


def perform_reranking(run, qfield, queries, docnos, doc_embs, word_dict, word_embs, sweep, normalizer, ref_measure, evaluator):
    """perform re-ranking of input run w/ semantic model"""
    # loop over weight values with sweep equal to sweep
    for weight in np.arange(0.0, 1.0, sweep):
        # generate combined run with current weight
        combined_run = compute_combined_run(run, qfield, queries, docnos, doc_embs, word_dict, word_embs, normalizer, weight)
        # evaluate combined run
        results = evaluator.evaluate(combined_run)
        # compute aggregated measure score
        agg_measure_score = pytrec_eval.compute_aggregated_measure(ref_measure, [qscore[ref_measure] for qscore in results.values()])
        # return aggregated mesure score and weight
        yield agg_measure_score, weight


def get_score(run, qrels, measure):
    """return np array of scores for a given measure"""
    if "P_" in measure:
        cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + " " + qrels + " " + run
    elif "ndcg_cut" in measure:
        cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
    else:
        cmd = "./trec_eval/trec_eval -q -m " + measure + " " + qrels + " " + run
    # run trev_eval as a subprocess
    process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    result = process.stdout.decode('utf-8').split('\n')
    # get scores
    scores = np.array([(score.split('\t')[-2], score.split('\t')[-1]) for score in result if score.split('\t')[0].strip() == measure and score.split('\t')[-2] != 'all'])
    return scores


def get_averaged_measure_score(run, qrels, measure):
    """return averaged measure score over topics"""
    if "P_" in measure:
        cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + " " + qrels + " " + run
    elif "ndcg_cut" in measure:
        cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
    else:
        cmd = "./trec_eval/trec_eval -m " + measure + " " + qrels + " " + run
    process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    result = process.stdout.decode('utf-8').split('\n')
    qscore = np.array([score.split('\t')[-1] for score in result if score.split('\t')[0].strip() == measure])
    qscore = qscore.astype(np.float)[0]
    return qscore


def get_averaged_inferred_measure_score(run, qrels, measure):
    """return averaged measure score over topics"""
    cmd = "perl sample_eval.pl " + qrels + " " + run
    process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    result = process.stdout.decode('utf-8').split('\n')
    qscore = np.array([score.split('\t')[-1] for score in result if score.split('\t')[0].strip() == measure])
    qscore = qscore.astype(np.float)[0]
    return qscore


def evaluate(measures, ranking_folder, ranking_name, qrels_folder, qrels_name):
    """evaluate models on given measures"""
    scores = []
    print('evaluate model ranking')
    if type(measures) == list:  # list of measures provided
        for measure in measures:
            scores.append(get_averaged_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measure)) 
    else:  # single measure provided
        scores.append(get_averaged_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measures))
    print("\t".join([measure + ": " + str(score) for measure, score in zip(measures, scores)]))
    return {measure: score for measure, score in zip(measures, scores)}


def evaluate_inferred(measures, ranking_folder, ranking_name, qrels_folder, qrels_name):
    """evaluate models on given inferred measures"""
    scores = []
    print('evaluate model ranking')
    if type(measures) == list:  # list of inferred measures provided
        for measure in measures:
            scores.append(get_averaged_inferred_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measure))
    else:  # single inferred measure provided
        scores.append(get_averaged_inferred_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measures))
    print("\t".join([measure + ": " + str(score) for measure, score in zip(measures, scores)]))
    return {measure: score for measure, score in zip(measures, scores)}
