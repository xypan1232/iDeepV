
iDeepV: predicting RBP binding sites using vector representaiton learned from sequences with CNNs.

In this study, we test iDeepV on two datasets:
RBP-31: download from https://github.com/xypan1232/iDeepS/tree/master/datasets/clip
RBP-24: download from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2

RNA2Vec.py is used to learn embdeddings for k-mers, and its input is the fasta file for training word2vec
iDpeepV.py is used to perform the predictions of RBP binding sites, the user only need input which dataset to test (RBP-24 or RBP-31). And reproduce the results in paper. 

# Dependency:
python 2.7 <br>
Keras 1.1.2: https://github.com/fchollet/keras <br>
genism: https://radimrehurek.com/gensim/models/word2vec.html <br>
sklearn v0.17: https://github.com/scikit-learn/scikit-learn <br>

# Reference
Xiaoyong Pan^, Hong-Bin Shen^. <a href="https://www.sciencedirect.com/science/article/pii/S0925231218304685">Learning distributed representations of RNA sequences and its application for predicting RNA-protein binding sites with a convolutional neural network</a>. Neurocomputing. 2018, 305: 51-58.
