#include "HuffmanEncoder.h"
#include <algorithm>
#include <assert.h>

HuffmanEncoder::HuffmanEncoder()
{
	m_dict = NULL;
}

void HuffmanEncoder::Save2File(const char* filename)
{
	FILE* fid = fopen(filename, "w");
	if(fid)
	{
		fprintf(fid, "%lld\n", m_hufflabel_info.size());

		for (unsigned i = 0; i < m_hufflabel_info.size(); ++i)
		{
			const auto& info = m_hufflabel_info[i];
			const auto& word = m_dict->GetWordInfo(i);
			fprintf(fid, "%s %d", word->word.c_str(), info.codelen);

			for (int j = 0; j < info.codelen; ++j)
				fprintf(fid, " %d", info.code[j]);

			for (int j = 0; j < info.codelen; ++j)
				fprintf(fid, " %d", info.point[j]);

			fprintf(fid, "\n");
		}

		fclose(fid);
	}
	else
	{
		printf("file open failed %s", filename);
	}
}

void HuffmanEncoder::RecoverFromFile(const char* filename)
{
	m_dict = new Dictionary();
	FILE* fid = fopen(filename, "r");
	if(fid)
	{
		int vocab_size;
		fscanf(fid, "%lld", &vocab_size);
		m_hufflabel_info.reserve(vocab_size);
		m_hufflabel_info.clear();

		int tmp;
		char sz_label[MAX_WORD_SIZE];
		for (int i = 0; i < vocab_size; ++i)
		{
			HuffLabelInfo info;

			fscanf(fid, "%s", sz_label, MAX_WORD_SIZE);
			m_dict->Insert(sz_label);
		
			fscanf(fid, "%d", &info.codelen);

			info.code.clear();
			info.point.clear();

			for (int j = 0; j < info.codelen; ++j)
			{
				fscanf(fid, "%d", &tmp);
				info.code.push_back(tmp);
			}
			for (int j = 0; j < info.codelen; ++j)
			{
				fscanf(fid, "%d", &tmp);
				info.point.push_back(tmp);
			}

			m_hufflabel_info.push_back(info);
		}
		fclose(fid);
	}
	else
	{
		printf("file open failed %s", filename);
	}
}

bool compare(const std::pair<int, int64_t>& x, const std::pair<int, int64_t>& y)
{
	if (x.second == 0) return true;
	if (y.second == 0) return false;
	return (x.second > y.second);
}

void HuffmanEncoder::BuildHuffmanTreeFromDict()
{
	std::vector<std::pair<int, int64_t> > ordered_words;
	ordered_words.reserve(m_dict->Size());
	ordered_words.clear();
	for (unsigned i = 0; i < static_cast<unsigned>(m_dict->Size()); ++i)
		ordered_words.push_back(std::pair<int, int64_t>(i, m_dict->GetWordInfo(i)->freq));
	std::sort(ordered_words.begin(), ordered_words.end(), compare);

	unsigned vocab_size = (unsigned) ordered_words.size();
	int64_t *count = new int64_t[vocab_size * 2 + 1]; //frequence
	unsigned *binary = new unsigned[vocab_size * 2 + 1]; //huffman code relative to parent node [1,0] of each node
	memset(binary, 0, sizeof(unsigned)* (vocab_size * 2 + 1));
	
	unsigned *parent_node = new unsigned[vocab_size * 2 + 1]; //
	memset(parent_node, 0, sizeof(unsigned)* (vocab_size * 2 + 1));
	unsigned code[MAX_CODE_LENGTH], point[MAX_CODE_LENGTH];

	for (unsigned i = 0; i < vocab_size; ++i)
		count[i] = ordered_words[i].second;
	for (unsigned i = vocab_size; i < vocab_size * 2; i++)
		count[i] = static_cast<int64_t>(1e15);
	int pos1 = vocab_size - 1;
	int pos2 = vocab_size;
	int min1i, min2i;
	for (unsigned i = 0; i < vocab_size - 1; i++)
	{
	    // First, find two smallest nodes 'min1, min2'
		assert(pos2 < vocab_size * 2 - 1);
		//find the samllest node
		if (pos1 >= 0) 
		{
			if (count[pos1] < count[pos2]) 
			{
				min1i = pos1;
				pos1--;
			} 
			else 
			{
				min1i = pos2;
				pos2++;
			}
		} 
		else
		{
			min1i = pos2;
			pos2++;
		}
		
		//find the second samllest node
		if (pos1 >= 0) 
		{
			if (count[pos1] < count[pos2]) 
			{
				min2i = pos1;
				pos1--;
			} 
			else 
			{
				min2i = pos2;
				pos2++;
			}
		} 
		else 
		{
			min2i = pos2;
			pos2++;
		}

		count[vocab_size + i] = count[min1i] + count[min2i];

		assert(min1i >= 0 && min1i < vocab_size * 2 - 1 && min2i >= 0 && min2i < vocab_size * 2 - 1);
		parent_node[min1i] = vocab_size + i;
		parent_node[min2i] = vocab_size + i;
		binary[min2i] = 1;
	}
	assert(pos1 < 0);
	
	//generate the huffman code for each leaf node
	m_hufflabel_info.clear();
	for (unsigned a = 0; a < vocab_size; ++a)
		m_hufflabel_info.push_back(HuffLabelInfo());
	for (unsigned a = 0; a < vocab_size; a++)
	{
		unsigned b = a, i = 0;
		while (1) 
		{
			assert(i < MAX_CODE_LENGTH);
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		unsigned cur_word = ordered_words[a].first;

		m_hufflabel_info[cur_word].codelen = i;
		m_hufflabel_info[cur_word].point.push_back(vocab_size - 2);

		for (b = 0; b < i; b++) 
		{
			m_hufflabel_info[cur_word].code.push_back(code[i - b - 1]);
			if (b)
				m_hufflabel_info[cur_word].point.push_back(point[i - b] - vocab_size);
		}
	}
	 
	delete[] count;
	count = nullptr;
	delete[] binary;
	binary = nullptr;
	delete[] parent_node;
	parent_node = nullptr;
}

void HuffmanEncoder::BuildFromTermFrequency(const char* filename)
{
	FILE* fid = fopen(filename, "r");
	if(fid)
	{
		char sz_label[MAX_WORD_SIZE];
		m_dict = new Dictionary();

		while (fscanf(fid, "%s", sz_label, MAX_WORD_SIZE) != EOF)
		{
			HuffLabelInfo info;
			int freq;
			fscanf(fid, "%d", &freq);
			m_dict->Insert(sz_label, freq);
		}
		fclose(fid);

		BuildHuffmanTreeFromDict();
	}
	else
	{
		printf("file open failed %s", filename);
	}
}

void HuffmanEncoder::BuildFromTermFrequency(Dictionary* dict)
{
	m_dict = dict;
	BuildHuffmanTreeFromDict();
}

int HuffmanEncoder::GetLabelSize()
{
	return m_dict->Size();
}

int HuffmanEncoder::GetLabelIdx(const char* label)
{
	return m_dict->GetWordIdx(label);
}

HuffLabelInfo* HuffmanEncoder::GetLabelInfo(char* label)
{
	int idx = GetLabelIdx(label);
	if (idx == -1)
		return NULL;
	return GetLabelInfo(idx);
}

HuffLabelInfo* HuffmanEncoder::GetLabelInfo(int label_idx)
{
	if (label_idx == -1) return NULL;
	return &m_hufflabel_info[label_idx];
}

Dictionary* HuffmanEncoder::GetDict()
{
	return m_dict;
}