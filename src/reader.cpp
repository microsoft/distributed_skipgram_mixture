#include "reader.h"

Reader::Reader(Dictionary *dictionary, Option *option)
{
	m_dictionary = dictionary;
	m_option = option;
	
	m_stopwords_table.clear();
	if (m_option->stopwords)
	{
		FILE* fid = fopen(m_option->sw_file, "r");
		while (ReadWord(m_word, fid))
		{
			m_stopwords_table.insert(m_word);
			if (m_dictionary->GetWordIdx(m_word) != -1)
				m_option->total_words -= m_dictionary->GetWordInfo(m_word)->freq;
		}

		fclose(fid);
	}
}

void Reader::Open(const char *input_file)
{
	m_fin = fopen(input_file, "r");
}

void Reader::Close()
{
	fclose(m_fin);
	m_fin = nullptr;
}

int Reader::GetSentence(int *sentence, int64_t &word_count)
{
	int length = 0, word_idx;
	word_count = 0;
	while (1)
	{
		if (!ReadWord(m_word, m_fin))
			break;
		word_idx = m_dictionary->GetWordIdx(m_word);
		if (word_idx == -1)
			continue;
		word_count++;
		if (m_option->stopwords && m_stopwords_table.count(m_word))
			continue;
		sentence[length++] = word_idx;
		if (length >= MAX_SENTENCE_LENGTH)
			break;
	}

	return length;
}


bool Reader::ReadWord(char *word, FILE *fin)
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
			else continue;
		}
		word[idx++] = ch;
		if (idx >= MAX_STRING - 1) idx--;   // Truncate too long words
	}
	word[idx] = 0;
	return idx != 0;
}
