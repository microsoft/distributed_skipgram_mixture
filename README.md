# distributed_skipgram_mixture
Distributed Skipgram Mixture for Multisense Word Embedding
==========

The DMWE tool is a parallelization of the Skip-Gram Mixture [2] algorithm on top of the DMTK parameter server. It provides an efficient "scaling to industry size" solution for multi sense word embedding.

For more details, please view our website http://ms-dmtk.azurewebsites.net/word2vec_multi.html.

Download 
----------
$ git clone https://github.com/Microsoft/distributed_skipgram_mixture 

Installation
----------

**Prerequisite**

DMWE is built on top of the DMTK parameter sever, therefore please download and build this project first.

**For Windows**

Download and build the dependence by opening windows\distributed_skipgram_mixture\distributed_skipgram_mixture.sln using Visual Studio 2013 and building all the projects.

**For Ubuntu (Tested on Ubuntu 12.04)**

Download and build the dependence by running $ sh scripts/build.sh. Modify the include and lib path in Makefile. Then run $ make all -j4.

HyperParameter Settings
----------
See parameters_settings.txt.
