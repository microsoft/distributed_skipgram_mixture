#include "multiverso_skipgram_mixture.h"
#include <algorithm>

MultiversoSkipGramMixture::MultiversoSkipGramMixture(Option *option, Dictionary *dictionary, HuffmanEncoder *huffman_encoder, Reader *reader)
{
	m_option = option;
	m_dictionary = dictionary;
	m_huffman_encoder = huffman_encoder;
	m_reader = reader;

	InitSenseCntInfo();
}

void MultiversoSkipGramMixture::InitSenseCntInfo()
{
	//First, determine #senses for words according to configuration parameters: top_N and top_ratio
	int threshold = (m_option->top_N ? std::min(m_option->top_N, m_dictionary->Size()) : m_dictionary->Size());
	threshold = static_cast<int>(std::min(static_cast<real>(m_option->top_ratio) * m_dictionary->Size(), static_cast<real>(threshold)));

	m_word_sense_info.total_senses_cnt = threshold * m_option->sense_num_multi + (m_dictionary->Size() - threshold);

	std::pair<int, int64_t>* wordlist = new std::pair<int, int64_t>[m_dictionary->Size() + 10];
	for (int i = 0; i < m_dictionary->Size(); ++i)
		wordlist[i] = std::pair<int, int64_t>(i, m_dictionary->GetWordInfo(i)->freq);

	std::sort(wordlist, wordlist + m_dictionary->Size(), [](std::pair<int, int64_t> a, std::pair<int, int64_t> b) {
		return a.second > b.second;
	});

	m_word_sense_info.word_sense_cnts_info.resize(m_dictionary->Size());

	for (int i = 0; i < threshold; ++i)
		m_word_sense_info.word_sense_cnts_info[wordlist[i].first] = m_option->sense_num_multi;
	for (int i = threshold; i < m_dictionary->Size(); ++i)
		m_word_sense_info.word_sense_cnts_info[wordlist[i].first] = 1;

	//Then, read words #sense info from the sense file
	if (m_option->sense_file) 
	{
		FILE* fid = fopen(m_option->sense_file, "r");
		char word[1000];
		while (fscanf(fid, "%s", word) != EOF)
		{
			int word_idx = m_dictionary->GetWordIdx(word);
			if (word_idx == -1)
				continue;
			if (m_word_sense_info.word_sense_cnts_info[word_idx] == 1)
			{
				m_word_sense_info.word_sense_cnts_info[word_idx] = m_option->sense_num_multi;
				m_word_sense_info.total_senses_cnt += (m_option->sense_num_multi - 1);
			}
		}
		fclose(fid);
	}

	//At last, point pointers to the right position
	m_word_sense_info.p_input_embedding.resize(m_dictionary->Size());
	int cnt = 0;
	m_word_sense_info.multi_senses_words_cnt = 0;

	for (int i = 0; i < m_dictionary->Size(); ++i) 
	{
		m_word_sense_info.p_input_embedding[i] = cnt;
		if (m_word_sense_info.word_sense_cnts_info[i] > 1)
			m_word_sense_info.p_wordidx2sense_idx[i] = m_word_sense_info.multi_senses_words_cnt++;
		cnt += m_word_sense_info.word_sense_cnts_info[i];
	}

	printf("Total senses:%d, total multiple mearning words:%d\n", m_word_sense_info.total_senses_cnt, m_word_sense_info.multi_senses_words_cnt);

}

void MultiversoSkipGramMixture::Train(int argc, char *argv[])
{
	multiverso::Barrier* barrier = new multiverso::Barrier(m_option->thread_cnt);

	printf("Inited barrier\n");

	SkipGramMixtureNeuralNetwork<real>* word2vector_neural_networks[2] = { new SkipGramMixtureNeuralNetwork<real>(m_option, m_huffman_encoder, &m_word_sense_info, m_dictionary, m_dictionary->Size()),
		new SkipGramMixtureNeuralNetwork<real>(m_option, m_huffman_encoder, &m_word_sense_info, m_dictionary, m_dictionary->Size()) };

	// Create Multiverso ParameterLoader and Trainers, 
	// start Multiverso environment
	printf("Initializing Multiverso ...\n");

	fflush(stdout);
	std::vector<multiverso::TrainerBase*> trainers;
	for (int i = 0; i < m_option->thread_cnt; ++i)
	{
		trainers.push_back(new Trainer<real>(i, m_option, (void**)word2vector_neural_networks, barrier, m_dictionary, &m_word_sense_info, m_huffman_encoder));
	}

	ParameterLoader<real> *parameter_loader = new ParameterLoader<real>(m_option, (void**)word2vector_neural_networks, &m_word_sense_info);
	multiverso::Config config;
	config.max_delay = m_option->max_delay;
	config.num_servers = m_option->num_servers;
	config.num_aggregator = m_option->num_aggregator;
	config.lock_option = static_cast<multiverso::LockOption>(m_option->lock_option);
	config.num_lock = m_option->num_lock;
	config.is_pipeline = m_option->pipline;

	fflush(stdout);

	multiverso::Multiverso::Init(trainers, parameter_loader, config, &argc, &argv);

	fflush(stdout);
	multiverso::Log::ResetLogFile("log.txt");
	m_process_id = multiverso::Multiverso::ProcessRank();
	PrepareMultiversoParameterTables(m_option, m_dictionary);
	
	printf("Start to train ...\n");
	TrainNeuralNetwork();
	printf("Rank %d Finish training\n", m_process_id);

	delete barrier;
	delete word2vector_neural_networks[0];
	delete word2vector_neural_networks[1];
	for (auto &trainer : trainers)
	{
		delete trainer;
	}
	delete parameter_loader;
	multiverso::Multiverso::Close();
}

void MultiversoSkipGramMixture::AddMultiversoParameterTable(multiverso::integer_t table_id, multiverso::integer_t rows,
	multiverso::integer_t cols, multiverso::Type type, multiverso::Format default_format)
{
	multiverso::Multiverso::AddServerTable(table_id, rows, cols, type, default_format);
	multiverso::Multiverso::AddCacheTable(table_id, rows, cols, type, default_format, 0);
	multiverso::Multiverso::AddAggregatorTable(table_id, rows, cols, type, default_format, 0);
}

void MultiversoSkipGramMixture::PrepareMultiversoParameterTables(Option *opt, Dictionary *dictionary)
{
	multiverso::Multiverso::BeginConfig();
	int proc_count = multiverso::Multiverso::TotalProcessCount();

	// create tables
	AddMultiversoParameterTable(kInputEmbeddingTableId, m_word_sense_info.total_senses_cnt, opt->embeding_size, multiverso::Type::Float, multiverso::Format::Dense);
	AddMultiversoParameterTable(kEmbeddingOutputTableId, dictionary->Size(), opt->embeding_size, multiverso::Type::Float, multiverso::Format::Dense);
	AddMultiversoParameterTable(kWordCountActualTableId, 1, 1, multiverso::Type::LongLong, multiverso::Format::Dense);
	AddMultiversoParameterTable(kWordSensePriorTableId, m_word_sense_info.multi_senses_words_cnt, m_option->sense_num_multi, multiverso::Type::Float, multiverso::Format::Dense);

	// initialize input embeddings
	for (int row = 0; row < m_word_sense_info.total_senses_cnt; ++row)
	{
		for (int col = 0; col < opt->embeding_size; ++col)
		{
			multiverso::Multiverso::AddToServer<real>(kInputEmbeddingTableId, row, col, static_cast<real>((static_cast<real>(rand()) / RAND_MAX - 0.5) / opt->embeding_size / proc_count));
		}
	}

	//initialize sense priors
	for (int row = 0; row < m_word_sense_info.multi_senses_words_cnt; ++row)
	{
		for (int col = 0; col < opt->sense_num_multi; ++col)
		{
			multiverso::Multiverso::AddToServer<real>(kWordSensePriorTableId, row, col, 
				static_cast<real>(m_option->store_multinomial ? 1.0 / m_option->sense_num_multi : log(1.0 / m_option->sense_num_multi)));
		}
	}
	multiverso::Multiverso::EndConfig();
}

//Load the sentences from train file, and store them in data_block
void MultiversoSkipGramMixture::LoadData(DataBlock *data_block, Reader *reader, int64_t size)
{
	data_block->ReleaseSentences();
	while (data_block->Size() < m_option->data_block_size)
	{
		int64_t word_count = 0;
		int *sentence = new (std::nothrow)int[MAX_SENTENCE_LENGTH + 2];
		assert(sentence != nullptr);
		int sentence_length = reader->GetSentence(sentence, word_count);
		if (sentence_length > 0)
		{
			data_block->Add(sentence, sentence_length, word_count, (uint64_t)rand() * 10000 + (uint64_t)rand());
		}
		else
		{
			//Reader read eof
			delete[] sentence;
			return;
		}
	}
}

void MultiversoSkipGramMixture::PushDataBlock(
	std::queue<DataBlock*> &datablock_queue, DataBlock* data_block)
{

	multiverso::Multiverso::PushDataBlock(data_block);

	datablock_queue.push(data_block);
	//limit the max size of total datablocks to avoid out of memory
	while (static_cast<int64_t>(datablock_queue.size()) > m_option->max_preload_blocks_cnt)
	{
		std::chrono::milliseconds dura(200);
		std::this_thread::sleep_for(dura);
		
		RemoveDoneDataBlock(datablock_queue);
	}
}

//Remove the datablock which has been delt by parameterloader and trainer
void MultiversoSkipGramMixture::RemoveDoneDataBlock(std::queue<DataBlock*> &datablock_queue)
{
	while (datablock_queue.empty() == false
		&& datablock_queue.front()->IsDone())
	{
		DataBlock *p_data_block = datablock_queue.front();
		datablock_queue.pop();
		delete p_data_block;
	}
}

void MultiversoSkipGramMixture::TrainNeuralNetwork()
{
	std::queue<DataBlock*>datablock_queue;
	int data_block_count = 0;

	multiverso::Multiverso::BeginTrain();

	for (int curr_epoch = 0; curr_epoch < m_option->epoch; ++curr_epoch)
	{
		m_reader->Open(m_option->train_file);
		while (1)
		{
			++data_block_count;
			DataBlock *data_block = new (std::nothrow)DataBlock();
			assert(data_block != nullptr);
			clock_t start = clock();
			LoadData(data_block, m_reader, m_option->data_block_size);
			if (data_block->Size() <= 0)
			{
				delete data_block;
				break;
			}
			multiverso::Log::Info("Rank%d Load%d^thDataBlockTime:%lfs\n", m_process_id, data_block_count,
				(clock() - start) / (double)CLOCKS_PER_SEC);
			multiverso::Multiverso::BeginClock();
			PushDataBlock(datablock_queue, data_block);
			multiverso::Multiverso::EndClock();
		}

		m_reader->Close();

		multiverso::Multiverso::BeginClock();

		DataBlock *output_data_block = new DataBlock(); //Add a special data_block for dumping model files
		output_data_block->AddTable(kInputEmbeddingTableId);
		output_data_block->AddTable(kEmbeddingOutputTableId);
		output_data_block->AddTable(kWordSensePriorTableId);
		output_data_block->SetEpochId(curr_epoch);

		++data_block_count;
		multiverso::Multiverso::PushDataBlock(output_data_block);
		multiverso::Multiverso::EndClock();
	}

	multiverso::Log::Info("Rank %d pushed %d blocks\n", multiverso::Multiverso::ProcessRank(), data_block_count);

	multiverso::Multiverso::EndTrain();

	//After EndTrain, all the datablock are done,
	//we remove all the datablocks
	RemoveDoneDataBlock(datablock_queue);
}


