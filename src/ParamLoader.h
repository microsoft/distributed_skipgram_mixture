#pragma once

#include <multiverso.h>
#include "DataBlock.h"
#include "MultiversoTablesId.h"
#include "Util.h"
#include "HuffmanEncoder.h"
#include "SkipGramMixtureNeuralNetwork.h"
#include "Log.h"


/*!
* \brief The class ParameterLoader preloads the parameters from multiverso server
*/
template<typename T>
class ParameterLoader : public multiverso::ParameterLoaderBase
{
public:
	ParameterLoader(Option *opt, void ** word2vector_neural_networks, WordSenseInfo* word_sense_info);
	/*!
	* \brief Request the parameters from multiverso server according to data_block
	* \param data_block stores the information of sentences
	*/
	void ParseAndRequest(multiverso::DataBlockBase* data_block) override;

private:
	int m_parse_and_request_count;
	Option* m_option;
	clock_t m_start_time;
	WordSenseInfo* m_words_sense_info;
	void ** m_sgmixture_neural_networks;
	FILE* m_log_file;
};

