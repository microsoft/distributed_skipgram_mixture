import os
import copy
import time
import random
import sys
import shutil
import subprocess

from subprocess import STDOUT

#Execute a subprocess with standard output
def execute(command):    
    popen = subprocess.Popen(command, stdout=subprocess.PIPE)
    lines_iterator = iter(popen.stdout.readline, b"")
    for line in lines_iterator:
        print(line) # yield line
		
#Parameter w.r.t. MPI
work_dir = 'Your Directory'
port = 'Your port number for MPI'
machinefile= 'Your host file for MPI'

#Parameter w.r.t. SG-Mixture Training
size = 50
train = 'Your Training File'
read_vocab = 'Your Vocab File'
binary = 2
init_learning_rate = 0.025
epoch = 1
window = 5
threads = 8
mincount = 5
EM_iteration = 1
momentum = 0.05

default_sense = 1
#Default number of senses for multi sense words
sense_num_multi = 5

'''
Three ways of specifying multi sense words, each with sense_num_multi prototypes:
1)Set top_n frequent words
2)Set top_ratio (lie between 0 to 1) frequent words
3)Write all these words into sense_file
'''
top_n = 0
top_ratio = 0
sense_file = 'Your Sense File, see sense_file.txt as an example'

#Output files
binary_embedding_file = 'emb.bin'
text_embedding_file = 'emb.txt'
huff_tree_file = 'huff.txt'
outputlayer_binary_file = 'emb_out.bin'
outputlayer_text_file = 'emb_out.txt'

preload_cnt = 5

#Number of sentences for each datablock.
#Warning: for wiki2014, set it to 50000, for clueweb09, set it to 750000. Other values are not tested.
data_block_size = 50000

#Warning: enable pipeline in multiverso will lead to some performance drop
pipline = 0

#Whether to store the multinomial parameters in its original form. If false, will store their log values instead.
multinomial = 0

mpi_args = '-port {0} -wdir {1} -machinefile {2} '.format(port, work_dir, machinefile)
sg_mixture_args  = ' -train_file {0} -binary_embedding_file {1} -text_embedding_file {2} -threads {3} -size {4} -binary {5} -epoch {6} -init_learning_rate {7} -min_count {8} -window {9} -momentum {12} -EM_iteration {13} -top_n {14} -top_ratio {14} -default_sense {16} -sense_num_multi {17} -huff_tree_file {18} -vocab_file {19} -outputlayer_binary_file {20} -outputlayer_text_file {21} -read_sense {22} -data_block_size {23} -is_pipline {24} -store_multinomial {25} -max_preload_size {26}'.format(train, binary_embedding_file, text_embedding_file, threads, size, binary, epoch, init_learning_rate, mincount, window, momentum, EM_iteration, top_n,  top_ratio, default_sense, sense_num_multi, huff_tree_file, read_vocab, outputlayer_binary_file, outputlayer_text_file, sense_file, data_block_size, pipline, multinomial, preload_cnt)

print mpi_args
print sg_mixture_args

#Execute MPI
proc = execute("mpiexec " + mpi_args + 'distributed_skipgram_mixture ' + sg_mixture_args)
