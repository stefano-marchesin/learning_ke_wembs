Repository to contain runs, data, code, and information about the paper ``Learning Methods for Knowledge-Enhanced Word Embeddings'' (submitted to ECIR 2020 reproducibility track) by M. Agosti, S. Marchesin, and G. Silvello 

**Analysis of the &alpha; and &beta; hyper-parameters:** 
 
The plots in Figure 1 show the sensitivity of the &alpha; and &beta; hyper-parameters for the joint learning and the modified retrofitting models. For each re-ranking combination, values of  &alpha;, &beta; ∈ {0.02; 0.04; 0.1; 0.3; 0.4; 0.5; 0.6; 0.7; 0.9; 1} are tested
using the best &gamma; from Table 1 and their behavior is compared with the behavior reported in the reference paper. Blue plots represent the behavior of the reproduced models as &alpha;,&beta; varies, whereas red plots represent the behavior of the original models.

 ![alt tag] (https://raw.github.com/stefano-marchesin/learning_ke_wembs/raw/master/figure/fig1-4.jpg)

In general, we observe smaller performance variations for the reproduced versions as &alpha; and &beta change. This is especially true for the re-ranking methods using the joint learning model, where the original versions present performance variations greater than 0.025 for some &alpha; and &beta; values. This indicates that the impact of the relational knowledge, injected into word embeddings during learning, is minimized when the embeddings are applied to a re-ranking scenario. In fact, relying on a BoW method to gather an initial pool of 1000 documents leaves candidate documents most affected by the vocabulary mismatch (i.e., relevant documents that do not contain query terms) undiscovered. Consequently, the potential of knowledge-enhanced word embeddings is not fully expressed in a re-ranking scenario.
