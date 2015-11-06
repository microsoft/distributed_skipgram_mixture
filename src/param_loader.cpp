#include "param_loader.h"

template<typename T>
ParameterLoader<T>::ParameterLoader(Option *option, void** word2vector_neural_networks, WordSenseInfo* word_sense_info)
{
	m_option = option;
	m_parse_and_request_count = 0;
	m_sgmixture_neural_networks = word2vector_neural_networks;
	m_log_file = fopen("parameter_loader.log", "w");
	m_words_sense_info = word_sense_info;
}

template<typename T>
void ParameterLoader<T>::ParseAndRequest(multiverso::DataBlockBase *data_block)
{
	if (m_parse_and_request_count == 0)
	{
		m_start_time = clock();
	}

	fprintf(m_log_file, "%lf\n", (clock() - m_start_time) / (double)CLOCKS_PER_SEC);
	multiverso::Log::Info("Rank %d ParameterLoader begin %d\n", multiverso::Multiverso::ProcessRank(), m_parse_and_request_count);
	DataBlock *data = reinterpret_cast<DataBlock*>(data_block);

	SkipGramMixtureNeuralNetwork<T>* sg_mixture_neural_network = reinterpret_cast<SkipGramMixtureNeuralNetwork<T>*>(m_sgmixture_neural_networks[m_parse_and_request_count % 2]);
	++m_parse_and_request_count;
	data->UpdateNextRandom();
	sg_mixture_neural_network->PrepareParmeter(data);

	std::vector<int>& input_layer_nodes = sg_mixture_neural_network->GetInputLayerNodes();
	std::vector<int>& output_layer_nodes = sg_mixture_neural_network->GetOutputLayerNodes();
	assert(sg_mixture_neural_network->status == 0);
	sg_mixture_neural_network->status = 1;

	for (int i = 0; i < input_layer_nodes.size(); ++i)
	{
		int word_id = input_layer_nodes[i];
		for (int j = 0; j < m_words_sense_info->word_sense_cnts_info[word_id]; ++j)
			RequestRow(kInputEmbeddingTableId, m_words_sense_info->p_input_embedding[word_id] + j);
	}

	for (int i = 0; i < output_layer_nodes.size(); ++i)
		RequestRow(kEmbeddingOutputTableId, output_layer_nodes[i]);

	RequestRow(kWordCountActualTableId, 0);

	for (int i = 0; i < input_layer_nodes.size(); ++i)
	{
		int word_id = input_layer_nodes[i];
		if (m_words_sense_info->word_sense_cnts_info[word_id] > 1)
			RequestRow(kWordSensePriorTableId, m_words_sense_info->p_wordidx2sense_idx[word_id]);
	}

	std::vector<int> & tables = data->GetTables();
	for (int i = 0; i < tables.size(); ++i)
		RequestTable(tables[i]);

	multiverso::Log::Info("Rank %d ParameterLoader finish %d\n", multiverso::Multiverso::ProcessRank(), m_parse_and_request_count - 1);
	fprintf(m_log_file, "%lf\n", (clock() - m_start_time) / (double)CLOCKS_PER_SEC);
	assert(sg_mixture_neural_network->status == 1);
	sg_mixture_neural_network->status = 2;
}

template class ParameterLoader<float>;
template class ParameterLoader<double>;