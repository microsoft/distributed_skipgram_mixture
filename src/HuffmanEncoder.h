#pragma once

#include "Dictionary.h"

const int MAX_CODE_LENGTH = 100;

struct HuffLabelInfo
{
	std::vector<int> point; //internal node ids in the code path
	std::vector<char> code; //huffman code
	int codelen;
	HuffLabelInfo()
	{
		codelen = 0;
		point.clear();
		code.clear();
	}
};

class HuffmanEncoder
{
public:
	HuffmanEncoder();
	
	void Save2File(const char* filename);
	void RecoverFromFile(const char* filename);
	void BuildFromTermFrequency(const char* filename);
	void BuildFromTermFrequency(Dictionary* dict);

	int GetLabelSize();
	int GetLabelIdx(const char* label);
	HuffLabelInfo* GetLabelInfo(char* label);
	HuffLabelInfo* GetLabelInfo(int label_idx);
	Dictionary* GetDict();

private:
	void BuildHuffmanTreeFromDict();
	std::vector<HuffLabelInfo> m_hufflabel_info;
	Dictionary* m_dict;
};