#include "data_block.h"

size_t DataBlock::Size()
{
	return m_sentences.size();
}

void DataBlock::Add(int *head, int sentence_length, int64_t word_count, uint64_t next_random)
{
	Sentence sentence(head, sentence_length, word_count, next_random);
	m_sentences.push_back(sentence);
}

void DataBlock::UpdateNextRandom()
{
	for (int i = 0; i < m_sentences.size(); ++i)
		m_sentences[i].next_random *= (uint64_t)rand();
}

void DataBlock::Get(int index, int* &head, int &sentence_length, int64_t &word_count, uint64_t &next_random)
{
	if (index >= 0 && index < m_sentences.size())
	{
		m_sentences[index].Get(head, sentence_length, word_count, next_random);
	}
	else
	{
		head = nullptr;
		sentence_length = 0;
		word_count = 0;
		next_random = 0;
	}
}

void DataBlock::ReleaseSentences()
{
	for (int i = 0; i < m_sentences.size(); ++i)
		delete m_sentences[i].head;
	m_sentences.clear();
}

void DataBlock::AddTable(int table_id)
{
	m_tables.push_back(table_id);
}

std::vector<int> & DataBlock::GetTables()
{
	return m_tables;
}

void DataBlock::SetEpochId(const int epoch_id)
{
	m_epoch_id = epoch_id;
}

int DataBlock::GetEpochId()
{
	return m_epoch_id;
}
