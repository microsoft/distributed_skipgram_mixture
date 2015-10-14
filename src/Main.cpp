#include <thread>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <multiverso.h>
#include <barrier.h>

#include "Dictionary.h"
#include "HuffmanEncoder.h"
#include "Util.h"
#include "Reader.h"
#include "MultiversoSkipGramMixture.h"
#include "ParamLoader.h"
#include "Trainer.h"
#include "SkipGramMixtureNeuralNetwork.h"

bool ReadWord(char *word, FILE *fin)
{
	int idx = 0;
	char ch;
	while (!feof(fin))
	{
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
		{
			if (idx > 0)
			{
				if (ch == '\n')
					ungetc(ch, fin);
				break;
			}

			if (ch == '\n')
			{
				strcpy(word, (char *)"</s>");
				return true;
			}
			else
			{
				continue;
			}
		}

		word[idx++] = ch;
		if (idx >= MAX_STRING - 1) idx--;   // Truncate too long words
	}

	word[idx] = 0;
	return idx > 0;
}

// Read the vocabulary file; create the dictionary and huffman_encoder according opt
int64_t LoadVocab(Option *opt, Dictionary *dictionary, HuffmanEncoder *huffman_encoder)
{
	int64_t total_words = 0;
	char word[MAX_STRING];
	FILE* fid = nullptr;
	printf("vocab_file %s\n", opt->read_vocab_file);
	if (opt->read_vocab_file != nullptr && strlen(opt->read_vocab_file) > 0)
	{
		printf("Begin to load vocabulary file [%s] ...\n", opt->read_vocab_file);
		fid = fopen(opt->read_vocab_file, "r");
		int word_freq;
		while (fscanf(fid, "%s %d", word, &word_freq) != EOF)
		{
			dictionary->Insert(word, word_freq);
		}
	}

	dictionary->RemoveWordsLessThan(opt->min_count);
	printf("Dictionary size: %d\n", dictionary->Size());
	total_words = 0;
	for (int i = 0; i < dictionary->Size(); ++i)
		total_words += dictionary->GetWordInfo(i)->freq;
	printf("Words in Corpus %I64d\n", total_words);
	huffman_encoder->BuildFromTermFrequency(dictionary);
	fclose(fid);

	return total_words;
}


int main(int argc, char *argv[])
{
	srand(static_cast<unsigned int>(time(NULL)));
	Option *option = new Option();
	Dictionary *dictionary = new Dictionary();
	HuffmanEncoder *huffman_encoder = new HuffmanEncoder();

	// Parse argument and store them in option
	option->ParseArgs(argc, argv);
	option->PrintArgs();
	if (!option->CheckArgs())
	{
		printf("Fatal error in arguments\n");
		return -1;
	}
	// Read the vocabulary file; create the dictionary and huffman_encoder according opt
	printf("Loading vocabulary ...\n");
	option->total_words = LoadVocab(option, dictionary, huffman_encoder);
	printf("Loaded vocabulary\n");
	fflush(stdout);

	Reader *reader = new Reader(dictionary, option);

	MultiversoSkipGramMixture *multiverso_word2vector = new MultiversoSkipGramMixture(option, dictionary, huffman_encoder, reader);

	fflush(stdout);

	multiverso_word2vector->Train(argc, argv);

	delete multiverso_word2vector;
	delete reader;
	delete huffman_encoder;
	delete dictionary;
	delete option;

	return 0;
}
