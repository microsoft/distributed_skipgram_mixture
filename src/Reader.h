#pragma once

#include "Util.h"
#include "Dictionary.h"
#include <mutex>
#include <unordered_set>

class Reader
{
public:
	Reader(Dictionary *dictionary, Option *option);
	void Open(const char *input_file);
	void Close();
	int GetSentence(int *sentence, int64_t &word_count);

private:
	Option* m_option;
	FILE* m_fin;
	char m_word[MAX_STRING + 1];
	Dictionary *m_dictionary;
	std::unordered_set<std::string> m_stopwords_table;

	bool ReadWord(char *word, FILE *fin);
};