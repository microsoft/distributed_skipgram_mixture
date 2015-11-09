Distributed Multisense Word Embedding
==========

The Distributed Multisense Word Embedding(DMWE) tool is a parallelization of the Skip-Gram Mixture [1] algorithm on top of the DMTK parameter server. It provides an efficient "scaling to industry size" solution for multi sense word embedding.

For more details, please view our website http://ms-dmtk.azurewebsites.net/word2vec_multi.html.

Download 
----------
$ git clone https://github.com/Microsoft/distributed_skipgram_mixture 

Build
----------

**Prerequisite**

DMWE is built on top of the DMTK parameter sever, therefore please download and build DMTK first (https://github.com/Microsoft/multiverso).

**For Windows**

Open windows\distributed_skipgram_mixture\distributed_skipgram_mixture.sln using Visual Studio 2013. Add the necessary include path (for example, the path for DMTK multiverso) and lib path. Then build the solution.

**For Ubuntu (Tested on Ubuntu 12.04)**

Download and build by running $ sh build.sh. Modify the include and lib path in Makefile. Then run $ make all -j4.

HyperParameter Settings
----------
See ./scripts/parameters_settings.txt and ./scripts/run.py.

Reference
----------
[1] Tian, F., Dai, H., Bian, J., Gao, B., Zhang, R., Chen, E., & Liu, T. Y. (2014). A probabilistic model for learning multi-prototype word embeddings. In Proceedings of COLING (pp. 151-160).
