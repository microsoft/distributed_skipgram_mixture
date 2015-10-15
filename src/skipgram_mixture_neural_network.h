#pragma once

#include <vector>

#include "util.h"
#include <multiverso.h>
#include "huffman_encoder.h"
#include "multiverso_skipgram_mixture.h"
#include "cstring"

enum class UpdateDirection
{
	UPDATE_INPUT,
	UPDATE_OUTPUT
};

template<typename T>
class SkipGramMixtureNeuralNetwork
{
public:
	T learning_rate;
	T sense_prior_momentum;

	int status;
	SkipGramMixtureNeuralNetwork(Option* option, HuffmanEncoder* huffmanEncoder, WordSenseInfo* word_sense_info, Dictionary* dic, int dicSize);
	~SkipGramMixtureNeuralNetwork();

	void Train(int* sentence, int sentence_length, T* gamma, T* fTable, T* input_backup);

	/*!
	* \brief Collect all the input words and output nodes in the data block
	*/
	void PrepareParmeter(DataBlock *data_block);

	std::vector<int>& GetInputLayerNodes();
	std::vector<int>& GetOutputLayerNodes();

	/*!
	* \brief Set the pointers to those local parameters
	*/
	void SetInputEmbeddingWeights(int input_node_id, T* ptr);
	void SetOutputEmbeddingWeights(int output_node_id, T* ptr);
	void SetSensePriorWeights(int input_node_id, T*ptr);
	void SetSensePriorParaWeights(int input_node_id, T* ptr);

	/*!
	* \brief Get the pointers to those locally updated parameters
	*/
	T* GetInputEmbeddingWeights(int input_node_id);
	T* GetEmbeddingOutputWeights(int output_node_id);
	T* GetSensePriorWeights(int input_node_id);
	T* GetSensePriorParaWeights(int input_node_id);

private:
	Option *m_option;
	Dictionary *m_dictionary;
	HuffmanEncoder *m_huffman_encoder;
	int m_dictionary_size;
	
	WordSenseInfo* m_word_sense_info;

	T** m_input_embedding_weights_ptr; //Points to every word's input embedding vector
	bool *m_seleted_input_embedding_weights;
	T** m_output_embedding_weights_ptr;  //Points to every huffman node's embedding vector
	bool *m_selected_output_embedding_weights;

	T** m_sense_priors_ptr; //Points to the multinomial parameters, if store_multinomial is set to zero.
	T** m_sense_priors_paras_ptr;//Points to sense prior parameters. If store_multinomial is zero, then it points to the log of multinomial, otherwise points to the multinomial parameters

	std::vector<int> m_input_layer_nodes;
	std::vector<int> m_output_layer_nodes;

	typedef void(SkipGramMixtureNeuralNetwork<T>::*FunctionType)(int input_node, std::vector<std::pair<int, int> >& output_nodes, void* v_gamma, void* v_fTable, void* v_input_backup);

	/*!
	* \brief Parse the needed parameter in a window
	*/
	void Parse(int feat, int word_idx, std::vector<std::pair<int, int> >& output_nodes);

	/*!
	* \brief Parse a sentence and deepen into two branchs
	* \one for TrainNN,the other one is for Parameter_parse&request
	*/
	void ParseSentence(int* sentence, int sentence_length, T* gamma, T* fTable, T* input_backup, FunctionType function);
	
	/*!
	* \brief Copy the input_nodes&output_nodes to WordEmbedding private set
	*/
	void DealPrepareParameter(int input_nodes, std::vector<std::pair<int, int> >& output_nodes, void* v_gamma, void* v_fTable, void* v_input_backup);

	/*!
	* \brief Train a window sample and update the
	* \input-embedding&output-embedding vectors
	* \param word_input represent the input words
	* \param output_nodes represent the ouput nodes on huffman tree, including the node index and path label
	* \param v_gamma is the temp memory to store the posterior probabilities of each sense
	* \param v_fTable is the temp memory to store the sigmoid value of inner product of input and output embeddings
	* \param v_input_backup stores the input embedding vectors as backup
	*/
	void TrainSample(int word_input, std::vector<std::pair<int, int> >& output_nodes, void* v_gamma, void* v_fTable, void* v_input_backup);
	
	/*!
	* \brief The E-step, estimate the posterior multinomial probabilities 
	* \param word_input represent the input words
	* \param output_nodes represent the ouput nodes on huffman tree, including the node index and path label
	* \param posterior represents the calculated posterior log likelihood  
	* \param estimation represents the calculated gammas (see the paper), that is, the softmax terms of posterior
	* \param sense_prior represents the parameters of sense prior probablities for each polysemous words
	* \param f_m is the temp memory to store the sigmoid value of inner products of input and output embeddings
	*/
	T Estimate_Gamma_m(int word_input, std::vector<std::pair<int, int> >& output_nodes, T* posterior, T* estimation, T* sense_prior, T* f_m);

	/*!
	* \brief The M step: update the embedding vectors to maximize the Q function
	* \param word_input represent the input words
	* \param output_nodes represent the ouput nodes on huffman tree, including the node index and path label
	* \param estimation represents the calculated gammas (see the paper), that is, the softmax terms of posterior
	* \param f_m is the temp memory to store the sigmoid value of inner products of input and output embeddings 
	* \param input_backup stores the input embedding vectors as backup
	* \param direction: update input vectors or output vectors
	*/
	void UpdateEmbeddings(int word_input, std::vector<std::pair<int, int> >& output_nodes, T* estimation, T* f_m, T* input_backup, UpdateDirection direction);

	/*!
	* \brief The M Step: update the sense prior probabilities to maximize the Q function
	* \param word_input represent the input words
	* \param curr_priors are the closed form values of the sense priors in this iteration
	*/
	void Maximize_Pi(int word_input, T* curr_priors);

	/*
	* \brief Record the input word so that parameter loader can be performed
	*/
	void AddInputLayerNode(int node_id);

	/*
	* \brief Record the huffman tree node so that parameter loader can be performed
	*/
	void AddOutputLayerNode(int node_id);
};
