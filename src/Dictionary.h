#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "Util.h"

const int MAX_WORD_SIZE = 901;

struct WordInfo
{
	std::string word;
	int64_t freq;
	WordInfo()
	{
		freq = 0;
		word.clear();
	}
	WordInfo(const std::string& _word, int64_t _freq)
	{
		word = _word;
		freq = _freq;
	}
};

class Dictionary
{
public:
	Dictionary();
	Dictionary(int i);
	void Clear();
	void SetWhiteList(const std::vector<std::string>& whitelist);
	void RemoveWordsLessThan(int64_t min_count);
	void MergeInfrequentWords(int64_t threshold);
	void Insert(const char* word, int64_t cnt = 1);
	void LoadFromFile(const char* filename);
	void LoadTriLetterFromFile(const char* filename, unsigned int min_cnt = 1, unsigned int letter_count = 3);
	int GetWordIdx(const char* word);
	const WordInfo* GetWordInfo(const char* word);
	const WordInfo* GetWordInfo(int word_idx);
	int Size();
	void StartIteration();
	bool HasMore();
	const WordInfo* Next();
	std::vector<WordInfo>::iterator Begin();
	std::vector<WordInfo>::iterator End();

private:
	int combine;
	std::vector<WordInfo> m_word_info;
	std::vector<WordInfo>::iterator m_word_iterator;
	std::unordered_map<std::string, int> m_word_idx_map;
	std::unordered_set<std::string> m_word_whitelist;
};