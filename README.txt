This is an introduction of the code developed for the Deep Clustering Network (DCN). Please direct your emails to 

Bo Yang, yang4173@umn.edu

if you have troubles running the code, or find any bugs. 

Here is the paper: arxiv: https://arxiv.org/pdf/1610.04794v1.pdf
Bo Yang, Xiao Fu, Nicholas D. Sidiropoulos and Mingyi Hong "Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering"

==============================================
Main files 

run_raw_mnist.py  : Script to reproduce our results on raw-MNIST dataset
multi_layer_km.py : Main file for defining the network, as well as various utility functions.

You can start running the code by e.g. (on Ubuntu)

$: ./run_raw_mnist.sh

--
More documentations can be found inside each of the above files.
--

==============================================
Data preparation

The data file should be named like 'something.pkl.gz', i.e., it should be pickled and compressed by gzip, using python code as follow:

"""
with gzip.open('something.pkl.gz', 'wb') as f:
    cPickle.dump([train_x, train_y], f, protocol = 0)
"""
where train_x and train_y are numpy ndarray with shape
train_x: (n_samples, n_features)
train_y: (n_samples, )

==============================================

Main difference compared to previous release
1) Included the dependent files
2) Included sample data files
3) Added theano environment flag in the .sh file
4) Cleaned up the repo to exclude unnecessary files

==============================================
Dependencies

Theano   
scikit-learn
numpy
scipy



