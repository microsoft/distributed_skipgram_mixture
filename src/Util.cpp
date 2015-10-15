#include "util.h"

Option::Option()
{
	train_file = NULL;
	read_vocab_file = NULL;
	binary_embedding_file = NULL;
	text_embedding_file = NULL;

	sw_file = NULL;
	output_binary = 2;
	embeding_size = 0;
	thread_cnt = 1;
	window_size = 5;
	min_count = 5;
	data_block_size = 100;
	init_learning_rate = static_cast<real>(0.025);
	epoch = 1;
	stopwords = false;
	total_words = 0;
	
	//multisense config
	store_multinomial = false;
	EM_iteration = 1;
	top_N = 0;
	top_ratio = static_cast<real>(0.1);
	sense_num_multi = 1;
	init_sense_prior_momentum = static_cast<real>(0.1);
	sense_file = NULL; 
	huff_tree_file = NULL;
	outputlayer_binary_file = NULL; 
	outputlayer_text_file = NULL;

	// multiverso config
	num_servers = 0;
	num_aggregator = 1;
	lock_option = 1;
	num_lock = 100;
	max_delay = 0;
}

void Option::ParseArgs(int argc, char* argv[])
{
	for (int i = 1; i < argc; i += 2)
	{
		if (strcmp(argv[i], "-size") == 0) embeding_size = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-train_file") == 0) train_file = argv[i + 1];
		if (strcmp(argv[i], "-vocab_file") == 0) read_vocab_file = argv[i + 1];
		if (strcmp(argv[i], "-binary") == 0) output_binary = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-init_learning_rate") == 0) init_learning_rate = static_cast<real>(atof(argv[i + 1]));
		if (strcmp(argv[i], "-binary_embedding_file") == 0) binary_embedding_file = argv[i + 1];
		if (strcmp(argv[i], "-text_embedding_file") == 0) text_embedding_file = argv[i + 1];
		if (strcmp(argv[i], "-window") == 0) window_size = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-data_block_size") == 0) data_block_size = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-threads") == 0) thread_cnt = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-min_count") == 0) min_count = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-epoch") == 0) epoch = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-stopwords") == 0) stopwords = atoi(argv[i + 1]) != 0;
		if (strcmp(argv[i], "-sw_file") == 0)  sw_file = argv[i + 1];
		if (strcmp(argv[i], "-num_servers") == 0) num_servers = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-num_aggregator") == 0) num_aggregator = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-lock_option") == 0) lock_option = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-num_lock") == 0) num_lock = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-max_delay") == 0) max_delay = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-max_preload_size") == 0) max_preload_blocks_cnt = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-is_pipline") == 0) pipline = atoi(argv[i + 1]) != 0;

		if (strcmp(argv[i], "-sense_num_multi") == 0) sense_num_multi = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-momentum") == 0) init_sense_prior_momentum = static_cast<real>(atof(argv[i + 1]));
		if (strcmp(argv[i], "-EM_iteration") == 0) EM_iteration = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-store_multinomial") == 0) store_multinomial = atoi(argv[i + 1]) != 0;
		if (strcmp(argv[i], "-top_n") == 0) top_N = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-top_ratio") == 0) top_ratio = static_cast<real>(atof(argv[i + 1]));
		if (strcmp(argv[i], "-read_sense") == 0) sense_file = argv[i + 1];
		if (strcmp(argv[i], "-huff_tree_file") == 0) huff_tree_file = argv[i + 1];
		if (strcmp(argv[i], "-outputlayer_binary_file") == 0) outputlayer_binary_file = argv[i + 1];
		if (strcmp(argv[i], "-outputlayer_text_file") == 0) outputlayer_text_file = argv[i + 1];
	}
}

void Option::PrintArgs()
{
	printf("train_file: %s\n", train_file);
	printf("read_vocab_file: %s\n", read_vocab_file);
	printf("binary_embedding_file: %s\n", binary_embedding_file);
	printf("sw_file: %s\n", sw_file);
	printf("output_binary: %d\n", output_binary);
	printf("stopwords: %d\n", stopwords);
	printf("embeding_size: %d\n", embeding_size);
	printf("thread_cnt: %d\n", thread_cnt);
	printf("window_size: %d\n", window_size);
	printf("min_count: %d\n", min_count);
	printf("epoch: %d\n", epoch);
	printf("total_words: %lld\n", total_words);
	printf("init_learning_rate: %lf\n", init_learning_rate);
	printf("data_block_size: %d\n", data_block_size);
	printf("pre_load_data_blocks: %d\n", max_preload_blocks_cnt);
	printf("num_servers: %d\n", num_servers);
	printf("num_aggregator: %d\n", num_aggregator);
	printf("lock_option: %d\n", lock_option);
	printf("num_lock: %d\n", num_lock);
	printf("max_delay: %d\n", max_delay);
	printf("is_pipline:%d\n", pipline);
	printf("top_ratio: %lf\n", top_ratio);
	printf("top_N: %d\n", top_N);
	printf("store_multinomial: %d\n", store_multinomial);
}

//Check whether the user defined arguments are valid
bool Option::CheckArgs()
{
	if (!Util::IsFileExist(train_file))
	{
		printf("Train corpus does not exist\n");
		return false;
	}

	if (!Util::IsFileExist(read_vocab_file))
	{
		printf("Vocab file does not exist\n");
		return false;
	}

	if (output_binary && (binary_embedding_file == NULL || outputlayer_binary_file == NULL))
	{
		printf("Binary output file name not specified\n");
		return false;
	}

	if (output_binary % 2 == 0 && (text_embedding_file == NULL || outputlayer_text_file == NULL))
	{
		printf("Text output file name not specified\n");
		return false;
	}

	if (huff_tree_file == NULL)
	{
		printf("Huffman tree file name not speficied\n");
		return false;
	}

	if (stopwords && !Util::IsFileExist(sw_file))
	{
		printf("Stop words file does not exist\n");
		return false;
	}

	if (init_sense_prior_momentum < -eps || init_sense_prior_momentum >= 1)
	{
		printf("Init momentum %.4f out of range, must lie between 0.0 and 1.0\n", init_sense_prior_momentum);
		return false;
	}

	if (top_ratio < -eps || top_ratio >= 1)
	{
		printf("Top ratio %.4f out of range, must lie between 0.0 and 1.0\n", init_sense_prior_momentum);
		return false;
	}

	if (sense_num_multi > MAX_SENSE_CNT)
	{
		printf("Sense number is too big, the maximum value is 50\n");
		return false;
	}

	if (fabs(static_cast<real>(max_delay)) > eps)
	{
		printf("Warning: better set max_delay to 0!\n");
	}

	return true;
}

bool Util::ValidF(const real &f)
{
	return f < 1 || f >= 1;
}
