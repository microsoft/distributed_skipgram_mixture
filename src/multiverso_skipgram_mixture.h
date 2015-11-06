#pragma once

#include <vector>
#include <ctime>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <multiverso/multiverso.h>
#include <multiverso/log.h>

#include "util.h"
#include "huffman_encoder.h"
#include "data_block.h"
#include "param_loader.h"
#include "trainer.h"
#include "reader.h"

class MultiversoSkipGramMixture
{
public:
	MultiversoSkipGramMixture(Option *option, Dictionary *dictionary, HuffmanEncoder *huffman_encoder, Reader *reader);

	void Train(int argc, char *argv[]);

private:
	int m_process_id;
	Option* m_option;
	Dictionary* m_dictionary;
	HuffmanEncoder* m_huffman_encoder;
	Reader* m_reader;

	WordSenseInfo m_word_sense_info;

	/*!
	* \brief Complete the train task with multiverso
	*/
	void TrainNeuralNetwork();


	/*!
	* \brief Create a new table in the multiverso
	*/
	void AddMultiversoParameterTable(multiverso::integer_t table_id, multiverso::integer_t rows,
		multiverso::integer_t cols, multiverso::Type type, multiverso::Format default_format);

	/*!
	* \brief Prepare parameter table in the multiverso
	*/
	void PrepareMultiversoParameterTables(Option *opt, Dictionary *dictionary);

	
	/*!
	* \brief Load data from train_file to datablock
	* \param datablock the datablock which needs to be assigned
	* \param reader some useful function for calling
	* \param size datablock limit byte size
	*/
	void LoadData(DataBlock *data_block, Reader *reader, int64_t size);

	/*!
	* \brief Push the datablock into the multiverso and datablock_queue
	*/
	void PushDataBlock(std::queue<DataBlock*> &datablock_queue, DataBlock* data_block);

	/*!
	* \brief Remove datablock which is finished by multiverso thread
	* \param datablock_queue store the pushed datablocks
	*/
	void RemoveDoneDataBlock(std::queue<DataBlock*> &datablock_queue);

	/*!
	* \brief Init the sense count info for all words
	*/
	void InitSenseCntInfo();
};


