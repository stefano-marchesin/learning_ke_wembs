# Learning Methods for Knowledge-Enhanced Word Embeddings

Repository to contain code and information for: 
<p align="center">
<b><i>Learning Methods for Knowledge-Enhanced Word Embeddings</i></b>
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
