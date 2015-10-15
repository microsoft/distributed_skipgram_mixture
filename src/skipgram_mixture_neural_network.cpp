#include "skipgram_mixture_neural_network.h"

template<typename T>
SkipGramMixtureNeuralNetwork<T>::SkipGramMixtureNeuralNetwork(Option* option, HuffmanEncoder* huffmanEncoder, WordSenseInfo* word_sense_info,  Dictionary* dic, int dicSize)
{
	status = 0;
	m_option = option;
	m_huffman_encoder = huffmanEncoder;
	m_word_sense_info = word_sense_info;
	m_dictionary_size = dicSize;
	m_dictionary = dic;

	m_input_embedding_weights_ptr = new T*[m_dictionary_size];
	m_sense_priors_ptr = new T*[m_dictionary_size];
	m_sense_priors_paras_ptr = new T*[m_dictionary_size];

	m_output_embedding_weights_ptr = new T*[m_dictionary_size];
	m_seleted_input_embedding_weights = new bool[m_dictionary_size];
	m_selected_output_embedding_weights = new bool[m_dictionary_size];
	assert(m_input_embedding_weights_ptr != nullptr);
	assert(m_output_embedding_weights_ptr != nullptr);
	assert(m_seleted_input_embedding_weights != nullptr);
	assert(m_selected_output_embedding_weights != nullptr);
	memset(m_seleted_input_embedding_weights, 0, sizeof(bool) * m_dictionary_size);
	memset(m_selected_output_embedding_weights, 0, sizeof(bool) * m_dictionary_size);
}

template<typename T>
SkipGramMixtureNeuralNetwork<T>::~SkipGramMixtureNeuralNetwork()
{
	delete m_input_embedding_weights_ptr;
	delete m_output_embedding_weights_ptr;
	delete m_sense_priors_ptr;
	delete m_sense_priors_paras_ptr;
	delete m_seleted_input_embedding_weights;
	delete m_selected_output_embedding_weights;
}

template<typename T>
void SkipGramMixtureNeuralNetwork<T>::Train(int* sentence, int sentence_length, T* gamma, T* fTable, T* input_backup)
{
	ParseSentence(sentence, sentence_length, gamma, fTable, input_backup, &SkipGramMixtureNeuralNetwork<T>::TrainSample);
}

template<typename T>
//The E - step, estimate the posterior multinomial probabilities
T SkipGramMixtureNeuralNetwork<T>::Estimate_Gamma_m(int word_input, std::vector<std::pair<int, int> >& output_nodes, T* posterior_ll, T* estimation, T* sense_prior, T* f_m)
{
	T* inputEmbedding = m_input_embedding_weights_ptr[word_input];
	T f, log_likelihood = 0;
	for (int sense_idx = 0; sense_idx < m_word_sense_info->word_sense_cnts_info[word_input]; ++sense_idx, inputEmbedding += m_option->embeding_size)
	{
		posterior_ll[sense_idx] = sense_prior[sense_idx] < eps ? MIN_LOG : log(sense_prior[sense_idx]); //posterior likelihood for each sense

		int64_t fidx = sense_idx * MAX_CODE_LENGTH;

		for (int d = 0; d < output_nodes.size(); ++d, fidx++)
		{
			f = Util::InnerProduct(inputEmbedding, m_output_embedding_weights_ptr[output_nodes[d].first], m_option->embeding_size);
			f = Util::Sigmoid(f);
			f_m[fidx] = f;
			if (output_nodes[d].second) //huffman code, 0 or 1
				f = 1 - f;
			posterior_ll[sense_idx] += f < eps ? MIN_LOG : log(f);
		}
		log_likelihood += posterior_ll[sense_idx];
	}
	if (m_word_sense_info->word_sense_cnts_info[word_input] == 1)
	{
		estimation[0] = 1;
		return log_likelihood;
	}

	Util::SoftMax(posterior_ll, estimation, m_word_sense_info->word_sense_cnts_info[word_input]);

	return log_likelihood;
}

template<typename T>
//The M Step: update the sense prior probabilities to maximize the Q function
void SkipGramMixtureNeuralNetwork<T>::Maximize_Pi(int word_input, T* log_likelihood)
{
	if (m_word_sense_info->word_sense_cnts_info[word_input] == 1)
	{
		return;
	}

	for (int sense_idx = 0; sense_idx < m_word_sense_info->word_sense_cnts_info[word_input]; ++sense_idx)
	{
		T new_alpha = log_likelihood[sense_idx];
		m_sense_priors_paras_ptr[word_input][sense_idx] = m_sense_priors_paras_ptr[word_input][sense_idx] * sense_prior_momentum + new_alpha * (1 - sense_prior_momentum);
	}

	if (!m_option->store_multinomial)
		Util::SoftMax(m_sense_priors_paras_ptr[word_input], m_sense_priors_ptr[word_input], m_option->sense_num_multi); //Update the multinomial parameters
}

template<typename T>
//The M step : update the embedding vectors to maximize the Q function
void SkipGramMixtureNeuralNetwork<T>::UpdateEmbeddings(int word_input, std::vector<std::pair<int, int> >& output_nodes, T* estimation, T* f_m, T* input_backup, UpdateDirection direction)
{
	T g;
	T* output_embedding;
	T* inputEmbedding;
	if (direction == UpdateDirection::UPDATE_INPUT)
		inputEmbedding = m_input_embedding_weights_ptr[word_input];
	else inputEmbedding = input_backup;
	for (int sense_idx = 0; sense_idx < m_word_sense_info->word_sense_cnts_info[word_input]; ++sense_idx, inputEmbedding += m_option->embeding_size)
	{
		int64_t fidx = sense_idx * MAX_CODE_LENGTH;
		for (int d = 0; d < output_nodes.size(); ++d, ++fidx)
		{
			output_embedding = m_output_embedding_weights_ptr[output_nodes[d].first];
			g = estimation[sense_idx] * (1 - output_nodes[d].second - f_m[fidx]) * learning_rate;
			if (direction == UpdateDirection::UPDATE_INPUT) //Update Input
			{
				for (int j = 0; j < m_option->embeding_size; ++j)
					inputEmbedding[j] += g * output_embedding[j];
			}
			else  // Update Output
			{
				for (int j = 0; j < m_option->embeding_size; ++j)
					output_embedding[j] += g * inputEmbedding[j];
			}
		}
	}
}


template<typename T>
//Train a window sample and update the input embedding & output embedding vectors
void SkipGramMixtureNeuralNetwork<T>::TrainSample(int input_node, std::vector<std::pair<int, int> >& output_nodes, void* v_gamma, void* v_fTable, void* v_input_backup)
{
	T* gamma = (T*)v_gamma; //stores the posterior probabilities
	T* fTable = (T*)v_fTable; //stores the inner product values of input and output embeddings
	T* input_backup = (T*)v_input_backup;

	T posterior_ll[MAX_SENSE_CNT]; //stores the posterior log likelihood
	T senses[1] = { 1.0 }; //For those words with only one sense

	T* sense_prior = m_word_sense_info->word_sense_cnts_info[input_node] == 1 ? senses : (m_option->store_multinomial ? m_sense_priors_paras_ptr[input_node] : m_sense_priors_ptr[input_node]);

	T log_likelihood;

	for (int iter = 0; iter < m_option->EM_iteration; ++iter)
	{
		// backup input embeddings
		memcpy(input_backup, m_input_embedding_weights_ptr[input_node], m_option->embeding_size * m_word_sense_info->word_sense_cnts_info[input_node] * sizeof(T));
		log_likelihood = 0;

		// E-Step
		log_likelihood += Estimate_Gamma_m(input_node, output_nodes, posterior_ll, gamma, sense_prior, fTable);

		// M-Step
		if (m_option->store_multinomial)
			Maximize_Pi(input_node, gamma);
		else
			Maximize_Pi(input_node, posterior_ll);

		UpdateEmbeddings(input_node, output_nodes, gamma, fTable, input_backup, UpdateDirection::UPDATE_INPUT);
		UpdateEmbeddings(input_node, output_nodes, gamma, fTable, input_backup, UpdateDirection::UPDATE_OUTPUT);

	}
}

template<typename T>
//Collect all the input words and output nodes in the data block
void SkipGramMixtureNeuralNetwork<T>::PrepareParmeter(DataBlock* data_block)
{
	for (int i = 0; i < m_input_layer_nodes.size(); ++i)
	{
		m_input_embedding_weights_ptr[m_input_layer_nodes[i]] = nullptr;
		m_seleted_input_embedding_weights[m_input_layer_nodes[i]] = false;
	}

	for (int i = 0; i < m_output_layer_nodes.size(); ++i)
	{
		m_output_embedding_weights_ptr[m_output_layer_nodes[i]] = nullptr;
		m_selected_output_embedding_weights[m_output_layer_nodes[i]] = false;
	}

	m_input_layer_nodes.clear();
	m_output_layer_nodes.clear();

	int sentence_length;
	int64_t word_count_deta;
	int* sentence;
	uint64_t next_random;

	for (int i = 0; i < data_block->Size(); ++i)
	{
		data_block->Get(i, sentence, sentence_length, word_count_deta, next_random);
		ParseSentence(sentence, sentence_length, nullptr, nullptr, nullptr, &SkipGramMixtureNeuralNetwork<T>::DealPrepareParameter);
	}
}

template<typename T>
//Copy the input_nodes&output_nodes to private set
void SkipGramMixtureNeuralNetwork<T>::DealPrepareParameter(int input_node, std::vector<std::pair<int, int> >& output_nodes, void* v_gamma, void* v_fTable, void* v_input_backup)
{
	AddInputLayerNode(input_node);
	for (int i = 0; i < output_nodes.size(); ++i)
		AddOutputLayerNode(output_nodes[i].first);
}

template<typename T>
/*
  Parse a sentence and deepen into two branchs:
  one for TrainNN, the other one is for Parameter_parse&request
*/
void SkipGramMixtureNeuralNetwork<T>::ParseSentence(int* sentence, int sentence_length, T* gamma, T* fTable, T* input_backup, FunctionType function)
{
	if (sentence_length == 0)
		return;

	int feat[MAX_SENTENCE_LENGTH + 10];
	int input_node;
	std::vector<std::pair<int, int> > output_nodes;
	for (int sentence_position = 0; sentence_position < sentence_length; ++sentence_position)
	{
		if (sentence[sentence_position] == -1) continue;
		int feat_size = 0;
		
		for (int i = 0; i < m_option->window_size * 2 + 1; ++i)
			if (i != m_option->window_size)
			{
				int c = sentence_position - m_option->window_size + i;
				if (c < 0 || c >= sentence_length || sentence[c] == -1) continue;
				feat[feat_size++] = sentence[c];

				//Begin: Train SkipGram
				{
					input_node = feat[feat_size - 1];
					output_nodes.clear();
					Parse(input_node, sentence[sentence_position], output_nodes);
					(this->*function)(input_node, output_nodes, gamma, fTable, input_backup);
				}
			}
	}
}

template<typename T>
//Parse the needed parameter in a window
void SkipGramMixtureNeuralNetwork<T>::Parse(int feat, int out_word_idx, std::vector<std::pair<int, int> >& output_nodes)
{
	const auto info = m_huffman_encoder->GetLabelInfo(out_word_idx);
	for (int d = 0; d < info->codelen; d++)
		output_nodes.push_back(std::make_pair(info->point[d], info->code[d]));

}

template<typename T>
void SkipGramMixtureNeuralNetwork<T>::AddInputLayerNode(int node_id)
{
	if (m_seleted_input_embedding_weights[node_id] == false)
	{
		m_seleted_input_embedding_weights[node_id] = true;
		m_input_layer_nodes.push_back(node_id);
	}
}

template<typename T>
void SkipGramMixtureNeuralNetwork<T>::AddOutputLayerNode(int node_id)
{
	if (m_selected_output_embedding_weights[node_id] == false)
	{
		m_selected_output_embedding_weights[node_id] = true;
		m_output_layer_nodes.push_back(node_id);
	}
}

template<typename T>
std::vector<int>& SkipGramMixtureNeuralNetwork<T>::GetInputLayerNodes()
{
	return m_input_layer_nodes;
}

template<typename T>
std::vector<int>& SkipGramMixtureNeuralNetwork<T>::GetOutputLayerNodes()
{
	return m_output_layer_nodes;
}

template<typename T>
void SkipGramMixtureNeuralNetwork<T>::SetInputEmbeddingWeights(int input_node_id, T* ptr)
{
	m_input_embedding_weights_ptr[input_node_id] = ptr;
}

template<typename T>
void SkipGramMixtureNeuralNetwork<T>::SetOutputEmbeddingWeights(int output_node_id, T* ptr)
{
	m_output_embedding_weights_ptr[output_node_id] = ptr;
}

template <typename T>
void SkipGramMixtureNeuralNetwork<T>::SetSensePriorWeights(int input_node_id, T*ptr)
{
	m_sense_priors_ptr[input_node_id] = ptr;
}

template <typename T>
void SkipGramMixtureNeuralNetwork<T>::SetSensePriorParaWeights(int input_node_id, T* ptr)
{
	m_sense_priors_paras_ptr[input_node_id] = ptr;
}

template<typename T>
T* SkipGramMixtureNeuralNetwork<T>::GetInputEmbeddingWeights(int input_node_id)
{
	return m_input_embedding_weights_ptr[input_node_id];
}

template<typename T>
T* SkipGramMixtureNeuralNetwork<T>::GetEmbeddingOutputWeights(int output_node_id)
{
	return m_output_embedding_weights_ptr[output_node_id];
}

template<typename T>
T* SkipGramMixtureNeuralNetwork<T>::GetSensePriorWeights(int input_node_id)
{
	return m_sense_priors_ptr[input_node_id];
}

template<typename T>
T* SkipGramMixtureNeuralNetwork<T>::GetSensePriorParaWeights(int input_node_id)
{
	return m_sense_priors_paras_ptr[input_node_id];
}

template class SkipGramMixtureNeuralNetwork<float>;
template class SkipGramMixtureNeuralNetwork<double>;
