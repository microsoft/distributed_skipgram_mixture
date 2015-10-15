#include "dictionary.h"

Dictionary::Dictionary()
{
	combine =0;
	Clear();
}

Dictionary::Dictionary(int i)
{
	combine = i;
	Clear();
}

void Dictionary::Clear()
{
	m_word_idx_map.clear();
	m_word_info.clear();
	m_word_whitelist.clear();
}

void Dictionary::SetWhiteList(const std::vector<std::string>& whitelist)
{
	for (unsigned int i = 0; i < whitelist.size(); ++i)
		m_word_whitelist.insert(whitelist[i]);
}

void Dictionary::MergeInfrequentWords(int64_t threshold)
{
	m_word_idx_map.clear();
	std::vector<WordInfo> tmp_info;
	tmp_info.clear();
	int infreq_idx = -1;

	for (auto& word_info : m_word_info)
	{
		if (word_info.freq >= threshold || word_info.freq == 0 || m_word_whitelist.count(word_info.word))
		{
			m_word_idx_map[word_info.word] = static_cast<int>(tmp_info.size());
			tmp_info.push_back(word_info);
		}
		else {
			if (infreq_idx < 0)
			{
				WordInfo infreq_word_info;
				infreq_word_info.word = "WE_ARE_THE_INFREQUENT_WORDS";
				infreq_word_info.freq = 0;
				m_word_idx_map[infreq_word_info.word] = static_cast<int>(tmp_info.size());
				infreq_idx = static_cast<int>(tmp_info.size());
				tmp_info.push_back(infreq_word_info);
			}
			m_word_idx_map[word_info.word] = infreq_idx;
			tmp_info[infreq_idx].freq += word_info.freq;
		}
	}
	m_word_info = tmp_info;
}

void Dictionary::RemoveWordsLessThan(int64_t min_count)
{
	m_word_idx_map.clear();
	std::vector<WordInfo> tmp_info;
	tmp_info.clear();
	for (auto& info : m_word_info)
	{
		if (info.freq >= min_count || info.freq == 0 || m_word_whitelist.count(info.word))
		{
			m_word_idx_map[info.word] = static_cast<int>(tmp_info.size());
			tmp_info.push_back(info);
		}
	}
	m_word_info = tmp_info;
}

void Dictionary::Insert(const char* word, int64_t cnt)
{
	const auto& it = m_word_idx_map.find(word);
	if (it != m_word_idx_map.end())
		m_word_info[it->second].freq += cnt;
	else 
	{
		m_word_idx_map[word] = static_cast<int>(m_word_info.size());
		m_word_info.push_back(WordInfo(word, cnt));
	}
}

void Dictionary::LoadFromFile(const char* filename)
{
	FILE* fid = fopen(filename, "r");

	if(fid)
	{
		char sz_label[MAX_WORD_SIZE];

		while (fscanf(fid, "%s", sz_label, MAX_WORD_SIZE) != EOF)
		{
			int freq;
			fscanf(fid, "%d", &freq);
			Insert(sz_label, freq);
		}
		fclose(fid);
	}
}

void Dictionary::LoadTriLetterFromFile(const char* filename, unsigned int min_cnt, unsigned int letter_count)
{
	FILE* fid = fopen(filename, "r");
	if(fid)
	{
		char sz_label[MAX_WORD_SIZE];
		while (fscanf(fid, "%s", sz_label, MAX_WORD_SIZE) != EOF)
		{
			int freq;
			fscanf(fid, "%d", &freq);
			if (static_cast<unsigned int>(freq) < min_cnt) continue;

			// Construct Tri-letter From word
			size_t len = strlen(sz_label);
			if (len > MAX_WORD_SIZE)
			{
				printf("ignore super long term");
				continue;
			}

			char tri_letters[MAX_WORD_SIZE + 2];
			tri_letters[0] = '#';
			int i = 0;
			for (i = 0; i < strlen(sz_label); i++) 
			{
				tri_letters[i+1] = sz_label[i];
			}

			tri_letters[i+1] = '#';
			tri_letters[i+2] = 0;
			if (combine) Insert(sz_label,freq);

			if (strlen(tri_letters) <= letter_count) {
				Insert(tri_letters, freq);
			} else {
				for (i = 0; i <= strlen(tri_letters) - letter_count; ++i) 
				{
					char tri_word[MAX_WORD_SIZE];
					unsigned int j = 0;
					for(j = 0; j < letter_count; j++)
					{
						tri_word[j] = tri_letters[i+j];
					}
					tri_word[j] = 0;
					Insert(tri_word, freq);
				}
			}
		}
		fclose(fid);
	}
}


int Dictionary::GetWordIdx(const char* word)
{
	const auto& it = m_word_idx_map.find(word);
	if (it != m_word_idx_map.end())
		return it->second;
	return -1;
}

int Dictionary::Size()
{
	return static_cast<int>(m_word_info.size());
}

const WordInfo* Dictionary::GetWordInfo(const char* word)
{
	const auto& it = m_word_idx_map.find(word);
	if (it != m_word_idx_map.end())
		return GetWordInfo(it->second);
	return NULL;
}
	
const WordInfo* Dictionary::GetWordInfo(int word_idx)
{
	if (word_idx >= 0 && word_idx < m_word_info.size())
		return &m_word_info[word_idx];
	return NULL;
}

void Dictionary::StartIteration() 
{
	m_word_iterator = m_word_info.begin();
}

bool Dictionary::HasMore() 
{
	return m_word_iterator != m_word_info.end();
}

const WordInfo* Dictionary::Next() 
{
	const WordInfo* entry = &(*m_word_iterator);
	++m_word_iterator;
	return entry;
}

std::vector<WordInfo>::iterator Dictionary::Begin()
{
	return m_word_info.begin();
}
std::vector<WordInfo>::iterator Dictionary::End()
{
	return m_word_info.end();
}
