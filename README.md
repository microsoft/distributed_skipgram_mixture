# distributed_skipgram_mixture
Distributed skipgram mixture model for multisense word embedding
The DMWE tool is a parallelization of the Skip-Gram Mixture [2] algorithm on top of the DMTK parameter server. It provides an efficient "scaling to industry size" solution for multi sense word embedding.

Why DMWE?

Word2vec [1] uses a single embedding vector for each word, which is not good enough to express the multiple meanings of polysemous words. To solve the problem, the Skip-Gram Mixture model was proposed to produce multiple embedding vectors for the polysemous words [2]. However, computing the multiple vectors for words are computationally expensive, and thus we develop the Distributed Multi-sense Word Embedding (DMWE) tool, which is highly scalable and efficient. It can be used to train multi-sense embedding vectors on very large scale dataset. The training process is powered by the DMTK framework: 
The DMTK parameter server stores the parameters in a distributed way, which means that each machine just holds a partition of the entire parameter set. This allows the entire embedding vector to be very large. For example, in our experiment on the ClueWeb data, the vocabulary size is 21 Million, and the number of parameters is up to over 2 billion. 
The training process in the clients is conducted in a streaming manner and is automatically pipelined. Specifically, during the training, the data are processed block by block. For each block, the client software will go the three step as aforementioned. The parameter request and model training steps in successive data blocks are pipelined so as to hide the delay caused by the network communication. Furthermore, in this way, the clients just need to hold the parameters for several data blocks simultaneously, corresponding to very economic memory usage. 
Downloading

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

Training on a single machine

Initialize the settings in the run.py according to your preference
Run run.py in the solution directory
Training with distributed setting

Using mpi:

Create a host.txt file containing all the machines to be used for training
Split your dataset into several parts and store them into the same directory of these machines
Distribute the same executable file into the same directory of these machines
Run the command line "smpd.exe -d -p port" in every machine
Run run.py in one of the machines with host.txt as its argument

Using ZMQ:

Compile the library of the DMTK parameter server, by specifying the communication mode to be ZMQ
Compile the project Multverso.Sever, and you will get the executable Multiverso.Sever.exe
Prepare a configuration file end_points.txt to describe the sever endpoints
Add a parameter setting in run.py, e.g.,'_endpoint_file=config.txt'
Start Multiverso.Sever.exe in each sever machine with appropriate command line arguments (please use Multiverso.Sever.exe -help for further information)
Execute run.py in one of the machines with end_points.txt as its argument
Algorithm configure for DMWE

For the Skip-Gram Mixture word embedding algorithms, we have provided hyperparemeters such as embedding size, number of polysemous words, number of senses and the others. You can specify their values in run.py

For the distributed training, users can configure the size of the data block, the mechanism for parameter update (such as ASP - Asynchronous Parallel, SSP - Stale Synchronous Parallel, BSP - Bulk Synchronous Parallel, and MA - Model Average), by setting the parameters in run.bat. For more details, please refer to the document of the DMTK parameter server.