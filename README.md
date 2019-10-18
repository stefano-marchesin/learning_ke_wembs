# Learning Methods for Knowledge-Enhanced Word Embeddings

Repository to contain code and information for the paper: 
<p align="center">
<b><i>Learning Methods for Knowledge-Enhanced Word Embeddings</i></b>
 </p>
Submitted to ECIR 2020 reproducibility track by M. Agosti, S. Marchesin, and G. Silvello 

## Additional Performance Analyses
**Analysis of the &alpha; and &beta; hyper-parameters for Joint Learning and Retrofitting Models:** 
 
The plots in Figure show the sensitivity of the &alpha; and &beta; hyper-parameters for the joint learning and the modified retrofitting models. For each re-ranking combination, values of  &alpha;, &beta; ∈ {0.02; 0.04; 0.1; 0.3; 0.4; 0.5; 0.6; 0.7; 0.9; 1} are tested
using the best &gamma; from Table 1 and their behavior is compared with the behavior reported in the reference paper. Blue plots represent the behavior of the reproduced models as &alpha;,&beta; varies, whereas red plots represent the behavior of the original models.

<p align="center">
<img src="https://raw.githubusercontent.com/stefano-marchesin/learning_ke_wembs/master/figure/fig1-4.jpg" alt="https://raw.githubusercontent.com/stefano-marchesin/learning_ke_wembs/master/figure/fig1-4.jpg" width="500" height="500">
</p>

In general, we observe smaller performance variations for the reproduced versions as &alpha; and &beta change. This is especially true for the re-ranking methods using the joint learning model, where the original versions present performance variations greater than 0.025 for some &alpha; and &beta; values. This indicates that the impact of the relational knowledge, injected into word embeddings during learning, is minimized when the embeddings are applied to a re-ranking scenario. In fact, relying on a BoW method to gather an initial pool of 1000 documents leaves candidate documents most affected by the vocabulary mismatch (i.e., relevant documents that do not contain query terms) undiscovered. Consequently, the potential of knowledge-enhanced word embeddings is not fully expressed in a re-ranking scenario.  

**Tukey's t-test to evaluate the statistical significance between the reproduced models:** 

<p align="center">
<img src="https://raw.githubusercontent.com/stefano-marchesin/learning_ke_wembs/master/figure/tukey_p10_OHSUMED.jpg" alt="https://raw.githubusercontent.com/stefano-marchesin/learning_ke_wembs/master/figure/tukey_p10_OHSUMED.jpg" width="400" height="500">
</p>

### Requirements

- ElasticSearch 6.6
- Python 3
  - Numpy
  - TensorFlow >= 1.13
  - Whoosh
  - SQLite3
  - Cvangysel
  - Pytrec_Eval
  - Scikit-Learn
  - Tqdm
  - QuickUMLS
  - Elasticsearch
  - Elasticsearch_dsl
- UMLS 2018AA

### Additional Notes
``server.py`` needs to be substitued within QuickUMLS folder as it contains a modified version required to run knowledge-enhanced models.  
The folder structure required to run experiments can be seen in folder ``example``. Python files need to be put in root.  
Qrels file needs to be in ``.txt`` format.  
To perform retrofitting models run ``retrofit_word_vecs.py``, to perform the alternate learning model run ``tf_run_jointcrm.py``, and to perform the joint learning model run ``tf_run_ccbow.py``.  
To run BM25 or QLM, use the Jupyter Notebook file ``elastic_search.ipynb``.  
To perform re-ranking run ``reranking.py``.
