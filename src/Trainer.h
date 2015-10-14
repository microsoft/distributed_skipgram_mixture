#pragma once

#include <thread>
#include <chrono>
#include <multiverso.h>
#include <log.h>
#include <barrier.h>

#include "DataBlock.h"
#include "MultiversoTablesId.h"
#include "Util.h"
#include "HuffmanEncoder.h"
#include "SkipGramMixtureNeuralNetwork.h"


template<typename T>
class Trainer : public multiverso::TrainerBase
{
public:
	Trainer(int trainer_id, Option *option, void** word2vector_neural_networks, multiverso::Barrier* barrier, Dictionary* dictionary, WordSenseInfo* word_sense_info, HuffmanEncoder* huff_encoder);
	
	/*!
	* /brief Train one datablock
	*/
	void TrainIteration(multiverso::DataBlockBase* data_block) override;

private:
	int m_process_id;
	int m_trainer_id;
	int m_train_count;	//threads count
	int m_process_count;  //machines count

	Option *m_option;
	WordSenseInfo* m_word_sense_info;
	HuffmanEncoder* m_huffman_encoder;

	int64_t m_word_count, m_last_word_count;
	
	T* gamma, * fTable, *input_backup; //temp memories to store middle results in the EM algorithm

	clock_t m_start_time, m_now_time, m_executive_time;
	void ** m_sgmixture_neural_networks;
	multiverso::Barrier *m_barrier;
	Dictionary* m_dictionary;
	FILE* m_log_file;

	/*!
	* \brief Save the multi sense input-embedding vectors
	* \param epoch_id, the embedding vectors after epoch_id is dumped
	*/
	void SaveMultiInputEmbedding(const int epoch_id);

	/*!
	* \brief Save the outpue embedding vectors, i.e. the embeddings for huffman tree nodes
	* \param epoch_id, the embedding vectors after epoch_id is dumped
	*/
	void SaveOutputEmbedding(const int epoch_id);

	/*!
	* \brief Save the Huffman tree structure
	*/
	void SaveHuffEncoder();

	/*!
	* \brief Copy the needed parameter from buffer to local blocks
	*/
	void CopyMemory(T* dest, multiverso::Row<T>& source, int size);
	int CopyParameterFromMultiverso(std::vector<int>& input_layer_nodes, std::vector<int>& output_layer_nodes, void* word2vector_neural_networks);
	
	/*!
	* \brief Add delta to the parameter stored in the
	* \buffer and send it to multiverso
	*/
	int AddParameterToMultiverso(std::vector<int>& input_layer_nodes, std::vector<int>& output_layer_nodes, void* word2vector_neural_networks);
	/*!
	* \brief Add delta of a row of local parameters to the parameter stored in the
	* \buffer and send it to multiverso
	* \param momentum: new_value = old_value * momentum + current_value * (1 - momentum). Set to non zero when updating the sense_priors
	*/
	void AddParameterRowToMultiverso(T* ptr, int table_id, int row_id, int size, real momentum = 0);

};

