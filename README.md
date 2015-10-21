# distributed_skipgram_mixture
Distributed skipgram mixture model for multisense word embedding
The DMWE tool is a parallelization of the Skip-Gram Mixture [2] algorithm on top of the DMTK parameter server. It provides an efficient "scaling to industry size" solution for multi sense word embedding.

To download the source codes of DMWE, please run
$ git clone https://github.com/Microsoft/distributed_skipgram_mixture 
Please note that DMWE is implemented in C++ for performance consideration.

Installation

Prerequisite

DMWE is built on top of the DMTK parameter sever, therefore please download and build this project first.

For Windows

Download and build the dependence by opening sln/Multiverso_Multi_Sense.sln using Visual Studio 2013 and building all the projects.

For Ubuntu (Tested on Ubuntu 12.04)

Download and build the dependence by running $ sh script/install_dep.sh. Modify the include and lib path in Makefile. Then run $ make all -j4.

Running DMWE

Initialize the settings in the run.py according to your preference.
Run run.py in the solution directory.
For the Skip-Gram Mixture word embedding algorithms, we have provided hyperparemeters such as embedding size, number of polysemous words, number of senses and the others. You can specify their values in run.py.
For the distributed training, users can configure the size of the data block, the mechanism for parameter update (such as ASP - Asynchronous Parallel, SSP - Stale Synchronous Parallel, BSP - Bulk Synchronous Parallel, and MA - Model Average), by setting the parameters in run.bat. For more details, please refer to the document of the DMTK parameter server.
The details of all the parameters in run.py are explained in parameters_setting.txt.