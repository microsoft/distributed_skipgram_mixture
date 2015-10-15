#pragma once

#include <fstream>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cassert>
#include <exception>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

typedef float real;

#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 2000
#define MAX_EXP 6
#define MAX_SENSE_CNT 50
#define MIN_LOG -15

const int table_size = (int)1e8;
const real eps = (real)1e-8;

struct WordSenseInfo
{
	std::vector<int> p_input_embedding; //Points to a word's row index in kInputEmbeddingTable
	std::unordered_map<int, int> p_wordidx2sense_idx; //Map a word's idx to its row index in the table kWordSensePriorTableId

	std::vector<int> word_sense_cnts_info; //Record every word's #sense count information
	int total_senses_cnt;
	int multi_senses_words_cnt; //Total number of words with multiple senses
};

struct Option
{
	const char* train_file;
	const char* read_vocab_file;
	const char* binary_embedding_file;
	const char* text_embedding_file;
	const char* sw_file;
	int output_binary, stopwords;
	int data_block_size;
	int embeding_size, thread_cnt, window_size, min_count, epoch;
	int64_t total_words;
	real init_learning_rate;
	int num_servers, num_aggregator, lock_option, num_lock, max_delay;
	bool pipline;
	int64_t max_preload_blocks_cnt;

	/*Multi sense config*/
	int EM_iteration;
	int top_N; //The top top_N frequent words has multi senses, e.g. 500, 1000,... 
	real top_ratio; // The top top_ratop frequent words has multi senses, e.g. 0.05, 0.1...
	int sense_num_multi; //Default number of senses for the multi_sense words
	real init_sense_prior_momentum;  //Initial momentum, momentum is used in updating the sense priors
	bool store_multinomial; //Use multinomial parameters. If set to false, use the log of multinomial instead
	const char* sense_file; //The sense file storing (word, #sense) mapping
	const char* huff_tree_file; // The output file storing the huffman tree structure
	const char* outputlayer_binary_file; //The output binary file storing all the output embedding(i.e. the huffman node embedding)
	const char* outputlayer_text_file; //The output text file storing all the output embedding(i.e. the huffman node embedding)

	Option();
	void ParseArgs(int argc, char* argv[]);
	void PrintArgs();
	bool CheckArgs();
};


class Util
{
public:
	static void SaveVocab();

	template<typename T>
	static T InnerProduct(T* x, T* y, int length)
	{
		T result = 0;
		for (int i = 0; i < length; ++i)
			result += x[i] * y[i];
		return result;
	}

	static bool ValidF(const real &f);

	template <typename T>
	static T Sigmoid(T f)
	{
		if (f < -MAX_EXP)
			return 0;
		if (f > MAX_EXP)
			return 1;
		return 1 / (1 + exp(-f));
	}

	template <typename T>
	static void SoftMax(T* s, T* result, int size)
	{
		T sum = 0, max_v = s[0];
		for (int j = 1; j < size; ++j)
			max_v = std::max(max_v, s[j]);
		for (int j = 0; j < size; ++j)
			sum += exp(s[j] - max_v);
		for (int j = 0; j < size; ++j)
			result[j] = exp(s[j] - max_v) / sum;
	}

	static bool IsFileExist(const char *fileName)
	{
		std::ifstream infile(fileName);
		return infile.good();
	}

};

