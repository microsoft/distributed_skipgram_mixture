#include "trainer.h"

template<typename T>
Trainer<T>::Trainer(int trainer_id, Option *option, void** word2vector_neural_networks, multiverso::Barrier *barrier, Dictionary* dictionary, WordSenseInfo* word_sense_info, HuffmanEncoder* huff_encoder)
{
	m_trainer_id = trainer_id;
	m_option = option;
	m_word_count = m_last_word_count = 0;
	m_sgmixture_neural_networks = word2vector_neural_networks;
	m_barrier = barrier;
	m_dictionary = dictionary;
	m_word_sense_info = word_sense_info;
	m_huffman_encoder = huff_encoder;

	gamma = (T*)calloc(m_option-> window_size * MAX_SENSE_CNT, sizeof(T));
	fTable = (T*)calloc(m_option-> window_size * MAX_CODE_LENGTH * MAX_SENSE_CNT, sizeof(T));
	input_backup = (T*)calloc(m_option->embeding_size * MAX_SENSE_CNT, sizeof(T));

	m_start_time = 0;
	m_train_count = 0;
	m_executive_time = 0;
	if (m_trainer_id == 0)
	{
		m_log_file = fopen("trainer.log", "w");
	}
}

template<typename T>
//Train one datablock
void Trainer<T>::TrainIteration(multiverso::DataBlockBase *data_block)
{
	if (m_train_count == 0)
	{
		m_start_time = clock();
		m_process_id = multiverso::Multiverso::ProcessRank();
	}

	printf("Rank %d Begin TrainIteration...%d\n", m_process_id, m_train_count);
	clock_t train_interation_start = clock();
	fflush(stdout);

	m_process_count = multiverso::Multiverso::TotalProcessCount();

	DataBlock *data = reinterpret_cast<DataBlock*>(data_block);
	SkipGramMixtureNeuralNetwork<T>* word2vector_neural_network = reinterpret_cast<SkipGramMixtureNeuralNetwork<T>*>(m_sgmixture_neural_networks[m_train_count % 2]);
	++m_train_count;
	std::vector<int>& input_layer_nodes = word2vector_neural_network->GetInputLayerNodes();
	std::vector<int>& output_layer_nodes = word2vector_neural_network->GetOutputLayerNodes();
	std::vector<int> local_input_layer_nodes, local_output_layer_nodes;
	assert(word2vector_neural_network->status == 2);
	if (m_trainer_id == 0)
	{
		multiverso::Log::Info("Rank %d input_layer_size=%d, output_layer_size=%d\n", m_process_id, input_layer_nodes.size(), output_layer_nodes.size());
	}

	for (int i = m_trainer_id; i < input_layer_nodes.size(); i += m_option->thread_cnt)
	{
		local_input_layer_nodes.push_back(input_layer_nodes[i]);
	}

	for (int i = m_trainer_id; i < output_layer_nodes.size(); i += m_option->thread_cnt)
	{
		local_output_layer_nodes.push_back(output_layer_nodes[i]);
	}
	
	CopyParameterFromMultiverso(local_input_layer_nodes, local_output_layer_nodes, word2vector_neural_network);

	multiverso::Row<int64_t>& word_count_actual_row = GetRow<int64_t>(kWordCountActualTableId, 0);
	T learning_rate = m_option->init_learning_rate * (1 - word_count_actual_row.At(0) / (T)(m_option->total_words * m_option->epoch + 1));
	if (learning_rate < m_option->init_learning_rate * (real)0.0001)
		learning_rate = m_option->init_learning_rate * (real)0.0001;
	word2vector_neural_network->learning_rate = learning_rate;

	//Linearly increase the momentum from init_sense_prior_momentum to 1
	word2vector_neural_network->sense_prior_momentum = m_option->init_sense_prior_momentum + 
		(1 - m_option->init_sense_prior_momentum) * word_count_actual_row.At(0) / (T)(m_option->total_words * m_option->epoch + 1);
	
	m_barrier->Wait();
	
	for (int i = m_trainer_id; i < data->Size(); i += m_option->thread_cnt)  //i iterates over all sentences
	{
		int sentence_length;
		int64_t word_count_deta;
		int *sentence;
		uint64_t next_random;
		data->Get(i, sentence, sentence_length, word_count_deta, next_random);

		word2vector_neural_network->Train(sentence, sentence_length, gamma, fTable, input_backup);
		
		m_word_count += word_count_deta;
		if (m_word_count - m_last_word_count > 10000)
		{
			multiverso::Row<int64_t>& word_count_actual_row = GetRow<int64_t>(kWordCountActualTableId, 0);
			Add<int64_t>(kWordCountActualTableId, 0, 0, m_word_count - m_last_word_count);
			m_last_word_count = m_word_count;
			m_now_time = clock();
			
			if (m_trainer_id % 3 == 0)
			{
				multiverso::Log::Info("Rank %d Trainer %d lr: %.5f Mom: %.4f Progress: %.2f%% Words/thread/sec(total): %.2fk  W/t/sec(executive): %.2fk\n",
					m_process_id, m_trainer_id,
					word2vector_neural_network->learning_rate, word2vector_neural_network->sense_prior_momentum,
					word_count_actual_row.At(0) / (real)(m_option->total_words * m_option->epoch + 1) * 100,
					m_last_word_count / ((real)(m_now_time - m_start_time + 1) / (real)CLOCKS_PER_SEC * 1000),
					m_last_word_count / ((real)(m_executive_time + clock() - train_interation_start + 1) / (real)CLOCKS_PER_SEC * 1000));

				fflush(stdout);
			}

			T learning_rate = m_option->init_learning_rate * (1 - word_count_actual_row.At(0) / (T)(m_option->total_words * m_option->epoch + 1));
			if (learning_rate < m_option->init_learning_rate * (real)0.0001)
				learning_rate = m_option->init_learning_rate * (real)0.0001;
			word2vector_neural_network->learning_rate = learning_rate;

			word2vector_neural_network->sense_prior_momentum = m_option->init_sense_prior_momentum + (1 - m_option->init_sense_prior_momentum) * word_count_actual_row.At(0) / (T)(m_option->total_words * m_option->epoch + 1);
		}
	}
	
	m_barrier->Wait();
	AddParameterToMultiverso(local_input_layer_nodes, local_output_layer_nodes, word2vector_neural_network);
	
	m_executive_time += clock() - train_interation_start;
	
	multiverso::Log::Info("Rank %d Train %d end at %lfs, cost %lfs, total cost %lfs\n",
		m_process_id,
		m_trainer_id, clock() / (double)CLOCKS_PER_SEC,
		(clock() - train_interation_start) / (double)CLOCKS_PER_SEC,
		m_executive_time / (double)CLOCKS_PER_SEC);
	fflush(stdout);

	if (data->GetTables().size() > 0 && m_trainer_id == 0) //Dump model files
	{
		SaveMultiInputEmbedding(data->GetEpochId());
		SaveOutputEmbedding(data->GetEpochId());
		if (data->GetEpochId() == 0)
			SaveHuffEncoder();

		fprintf(m_log_file, "%d %lf\t %lf\n", data->GetEpochId(), (clock() - m_start_time) / (double)CLOCKS_PER_SEC, m_executive_time / (double)CLOCKS_PER_SEC);
	}

	assert(word2vector_neural_network->status == 2);

	word2vector_neural_network->status = 0;

	multiverso::Log::Info("Rank %d Train %d are leaving training iter with nn status:%d\n", m_process_id, m_trainer_id, word2vector_neural_network->status);
	fflush(stdout);
}

template<typename T>
//Copy a size of memory from source row to dest
void Trainer<T>::CopyMemory(T* dest, multiverso::Row<T>& source, int size)
{
	for (int i = 0; i < size; ++i)
		dest[i] = source.At(i);
}

template<typename T>
//Copy the needed parameter from buffer to local blocks
int Trainer<T>::CopyParameterFromMultiverso(std::vector<int>& input_layer_nodes, std::vector<int>& output_layer_nodes, void* local_word2vector_neural_network)
{
	SkipGramMixtureNeuralNetwork<T>* word2vector_neural_network = (SkipGramMixtureNeuralNetwork<T>*)local_word2vector_neural_network;
	
	//Copy input embedding
	for (int i = 0; i < input_layer_nodes.size(); ++i)
	{
		T* ptr = (T*)calloc(m_word_sense_info->word_sense_cnts_info[input_layer_nodes[i]] * m_option->embeding_size, sizeof(T));
		int row_id_base = m_word_sense_info->p_input_embedding[input_layer_nodes[i]];
		for (int j = 0, row_id = row_id_base; j < m_word_sense_info->word_sense_cnts_info[input_layer_nodes[i]]; ++j, ++row_id)
			CopyMemory(ptr + j * m_option->embeding_size, GetRow<T>(kInputEmbeddingTableId, row_id), m_option->embeding_size);
		word2vector_neural_network->SetInputEmbeddingWeights(input_layer_nodes[i], ptr);
	}
	
	//Copy output embedding
	for (int i = 0; i < output_layer_nodes.size(); ++i)
	{
		T* ptr = (T*)calloc(m_option->embeding_size, sizeof(T));
		CopyMemory(ptr, GetRow<T>(kEmbeddingOutputTableId, output_layer_nodes[i]), m_option->embeding_size);
		for (int j = 0; j < m_option->embeding_size; j += 5)
			if (!Util::ValidF(static_cast<real>(ptr[j])))
			{
				printf("invalid number\n");
				fflush(stdout);
				throw std::runtime_error("Invalid output embeddings");
			}
		word2vector_neural_network->SetOutputEmbeddingWeights(output_layer_nodes[i], ptr);
	}
	
	//Copy sense prior
	for (int i = 0; i < input_layer_nodes.size(); ++i)
	{
		if (m_word_sense_info->word_sense_cnts_info[input_layer_nodes[i]] > 1)
		{
			T* ptr = (T*)calloc(m_option->sense_num_multi, sizeof(T));
			T* para_ptr = (T*)calloc(m_option->sense_num_multi, sizeof(T));

			CopyMemory(para_ptr, GetRow<T>(kWordSensePriorTableId, m_word_sense_info->p_wordidx2sense_idx[input_layer_nodes[i]]), m_option->sense_num_multi);

			if (!m_option->store_multinomial)//softmax the para_ptr to obtain the multinomial parameters
				Util::SoftMax(para_ptr, ptr, m_option->sense_num_multi);
			word2vector_neural_network->SetSensePriorWeights(input_layer_nodes[i], ptr);
			word2vector_neural_network->SetSensePriorParaWeights(input_layer_nodes[i], para_ptr);
		}
	}

	return 0;
}

template<typename T>
//Add delta of a row of local parameters to the parameter stored in the buffer and send it to multiverso
void Trainer<T>::AddParameterRowToMultiverso(T* ptr, int table_id, int row_id, int size, real momentum)
{
	multiverso::Row<T>& row = GetRow<T>(table_id, row_id);
	for (int i = 0; i < size; ++i)
	{
		T dest = ptr[i] * (1 - momentum) + row.At(i) * momentum;
		T delta = (dest - row.At(i)) / m_process_count;
		Add<T>(table_id, row_id, i, delta);
	}
}

template<typename T>
//Add delta to the parameter stored in the buffer and send it to multiverso
int Trainer<T>::AddParameterToMultiverso(std::vector<int>& input_layer_nodes, std::vector<int>& output_layer_nodes, void* local_word2vector_neural_network)
{
	SkipGramMixtureNeuralNetwork<T>* word2vector_neural_network = (SkipGramMixtureNeuralNetwork<T>*)local_word2vector_neural_network;
	std::vector<T*> blocks; //used to store locally malloced memorys

	//Add input embeddings
	for (int i = 0; i < input_layer_nodes.size(); ++i)
	{
		int table_id = kInputEmbeddingTableId;
		int row_id_base = m_word_sense_info->p_input_embedding[input_layer_nodes[i]];
		T* ptr = word2vector_neural_network->GetInputEmbeddingWeights(input_layer_nodes[i]);

		for (int j = 0, row_id = row_id_base; j < m_word_sense_info->word_sense_cnts_info[input_layer_nodes[i]]; ++j, ++row_id)
			AddParameterRowToMultiverso(ptr + m_option->embeding_size * j, table_id, row_id, m_option->embeding_size);
		blocks.push_back(ptr);
	}

	//Add output embeddings
	for (int i = 0; i < output_layer_nodes.size(); ++i)
	{
		int table_id = kEmbeddingOutputTableId;
		int row_id = output_layer_nodes[i];
		T* ptr = word2vector_neural_network->GetEmbeddingOutputWeights(row_id);
		AddParameterRowToMultiverso(ptr, table_id, row_id, m_option->embeding_size);
		blocks.push_back(ptr);
	}

	//Add sense priors
	for (int i = 0; i < input_layer_nodes.size(); ++i)
	{
		if (m_word_sense_info->word_sense_cnts_info[input_layer_nodes[i]] > 1)
		{
			int table_id = kWordSensePriorTableId;
			int row_id = m_word_sense_info->p_wordidx2sense_idx[input_layer_nodes[i]];

			T* ptr = word2vector_neural_network->GetSensePriorWeights(input_layer_nodes[i]);
			T* para_ptr = word2vector_neural_network->GetSensePriorParaWeights(input_layer_nodes[i]);

			AddParameterRowToMultiverso(para_ptr, table_id, row_id, m_option->sense_num_multi, static_cast<real>(word2vector_neural_network->sense_prior_momentum));

			blocks.push_back(ptr);
			blocks.push_back(para_ptr);
		}

	}

	for (auto& x : blocks)
		free(x);

	return 0;
}

template<typename T>
void Trainer<T>::SaveMultiInputEmbedding(const int epoch_id)
{
	FILE* fid = nullptr;
	T* sense_priors_ptr = (T*)calloc(m_option->sense_num_multi, sizeof(real));

	char outfile[2000];
	if (m_option->output_binary)
	{
		sprintf(outfile, "%s%d", m_option->binary_embedding_file, epoch_id);

		fid = fopen(outfile, "wb");

	    fprintf(fid, "%d %d %d\n", m_dictionary->Size(), m_word_sense_info->total_senses_cnt, m_option->embeding_size);
		for (int i = 0; i < m_dictionary->Size(); ++i)
		{
			fprintf(fid, "%s %d ", m_dictionary->GetWordInfo(i)->word.c_str(), m_word_sense_info->word_sense_cnts_info[i]);
			int emb_row_id;
			real emb_tmp;

			if (m_word_sense_info->word_sense_cnts_info[i] > 1)
			{
				CopyMemory(sense_priors_ptr, GetRow<T>(kWordSensePriorTableId, m_word_sense_info->p_wordidx2sense_idx[i]), m_option->sense_num_multi);
				if (!m_option->store_multinomial)
					Util::SoftMax(sense_priors_ptr, sense_priors_ptr, m_option->sense_num_multi);
				
				for (int j = 0; j < m_option->sense_num_multi; ++j)
				{
					fwrite(sense_priors_ptr + j, sizeof(real), 1, fid);
					emb_row_id = m_word_sense_info->p_input_embedding[i] + j;
					multiverso::Row<real>& embedding = GetRow<real>(kInputEmbeddingTableId, emb_row_id);
					for (int k = 0; k < m_option->embeding_size; ++k)
					{
						emb_tmp = embedding.At(k);
						fwrite(&emb_tmp, sizeof(real), 1, fid);
					}
				}
				fprintf(fid, "\n");
			}
			else
			{
				real prob = 1.0;
				fwrite(&prob, sizeof(real), 1, fid);
				emb_row_id = m_word_sense_info->p_input_embedding[i];
				multiverso::Row<real>& embedding = GetRow<real>(kInputEmbeddingTableId, emb_row_id);
				
				for (int k = 0; k < m_option->embeding_size; ++k)
				{
					emb_tmp = embedding.At(k);
					fwrite(&emb_tmp, sizeof(real), 1, fid);
				}
				fprintf(fid, "\n");
			}
		}

		fclose(fid);
	}
	if (m_option->output_binary % 2 == 0)
	{
		sprintf(outfile, "%s%d", m_option->text_embedding_file, epoch_id);

		fid = fopen(outfile, "w");
		fprintf(fid, "%d %d %d\n", m_dictionary->Size(), m_word_sense_info->total_senses_cnt, m_option->embeding_size);
		for (int i = 0; i < m_dictionary->Size(); ++i)
		{
			fprintf(fid, "%s %d\n", m_dictionary->GetWordInfo(i)->word.c_str(), m_word_sense_info->word_sense_cnts_info[i]);

			int emb_row_id;
			real emb_tmp;

			if (m_word_sense_info->word_sense_cnts_info[i] > 1)
			{
				CopyMemory(sense_priors_ptr, GetRow<T>(kWordSensePriorTableId, m_word_sense_info->p_wordidx2sense_idx[i]), m_option->sense_num_multi);

				if (!m_option->store_multinomial)
					Util::SoftMax(sense_priors_ptr, sense_priors_ptr, m_option->sense_num_multi);

				for (int j = 0; j < m_option->sense_num_multi; ++j)
				{
					fprintf(fid, "%.4f", sense_priors_ptr[j]);

					emb_row_id = m_word_sense_info->p_input_embedding[i] + j;
					multiverso::Row<real>& embedding = GetRow<real>(kInputEmbeddingTableId, emb_row_id);
					for (int k = 0; k < m_option->embeding_size; ++k)
					{
						emb_tmp = embedding.At(k);
						fprintf(fid, " %.3f", emb_tmp);
					}
					fprintf(fid, "\n");
				}
			}
			else
			{
				real prob = 1.0;
				fprintf(fid, "%.4f", 1.0);

				emb_row_id = m_word_sense_info->p_input_embedding[i];
				multiverso::Row<real>& embedding = GetRow<real>(kInputEmbeddingTableId, emb_row_id);
				for (int k = 0; k < m_option->embeding_size; ++k)
				{
					emb_tmp = embedding.At(k);
					fprintf(fid, " %.3f", emb_tmp);
				}
				fprintf(fid, "\n");
			}
		}

		fclose(fid);
	}
}

template<typename T>
void Trainer<T>::SaveOutputEmbedding(const int epoch_id)
{
	char outfile[2000];
	if (m_option->output_binary)
	{
		sprintf(outfile, "%s%d", m_option->outputlayer_binary_file, epoch_id);

		FILE* fid = fopen(outfile, "wb");
		fprintf(fid, "%d %d\n", m_dictionary->Size(), m_option->embeding_size);
		for (int i = 0; i < m_dictionary->Size(); ++i)
		{
			multiverso::Row<real>& hs_embedding = GetRow<real>(kEmbeddingOutputTableId, i);
			for (int j = 0; j < m_option->embeding_size; ++j)
			{
				real emb_tmp = hs_embedding.At(j);
				fwrite(&emb_tmp, sizeof(real), 1, fid);
			}
		}
		fclose(fid);
	}
	if (m_option->output_binary % 2 == 0)
	{
		sprintf(outfile, "%s%d", m_option->outputlayer_text_file, epoch_id);

		FILE* fid = fopen(outfile, "w");
		fprintf(fid, "%d %d\n", m_dictionary->Size(), m_option->embeding_size);
		for (int i = 0; i < m_dictionary->Size(); ++i)
		{
			multiverso::Row<real>& hs_embedding = GetRow<real>(kEmbeddingOutputTableId, i);

			for (int j = 0; j < m_option->embeding_size; ++j)
				fprintf(fid, "%.2f ", hs_embedding.At(j));
			fprintf(fid, "\n");
		}
		fclose(fid);
	}
}

template<typename T>
void Trainer<T>::SaveHuffEncoder()
{
	FILE* fid = fopen(m_option->huff_tree_file, "w");
	fprintf(fid, "%d\n", m_dictionary->Size());
	for (int i = 0; i < m_dictionary->Size(); ++i)
	{
		fprintf(fid, "%s", m_dictionary->GetWordInfo(i)->word.c_str());
		const auto info = m_huffman_encoder->GetLabelInfo(i);
		fprintf(fid, " %d", info->codelen);
		for (int j = 0; j < info->codelen; ++j)
			fprintf(fid, " %d", info->code[j]);
		for (int j = 0; j < info->codelen; ++j)
			fprintf(fid, " %d", info->point[j]);
		fprintf(fid, "\n");
	}
	fclose(fid);
}

template class Trainer<float>;
template class Trainer<double>;
