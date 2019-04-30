import os
import numpy as np
import nltk
import random

import torch
from torch.utils.data.dataset import Dataset

class Dictionary():
	def __init__(self, stopwords=[]):
		self.word2idx = {}
		self.words = []
		if len(stopwords) != 0:
			self.stopwords = stopwords.copy()

	def add_word(self, word):
		if word not in self.words and word not in self.stopwords:
			self.words.append(word)
			self.word2idx[word] = len(self.words) - 1

	def __len__(self): return len(self.words)

class Corpus():
	def __init__(self, data=[], dictionary_word =[], stopwords=[]):
		self.dictionary = Dictionary(stopwords)

		if len(dictionary_word) != 0:
			self.dictionary.words = dictionary_word.copy()

		for d in data:
			tokens = self.tokenize(d['comment'])
			for token in tokens: self.dictionary.add_word(token)

	def tokenize(self, sent):
		word_tokens = nltk.word_tokenize(sent)
		tokens = [w.lower() for w in word_tokens if w.isalnum()]
		return tokens

class CommentDataset(Dataset):
	def __init__(self, dataset, sent_size, corpus, dtype=""):
		data_pos = []
		data_neg = []
		for d in dataset:
			if d['star'] == 1: 
				data_neg.append(d['comment'])
			elif d['star'] == 5: 
				data_pos.append(d['comment'])

		if dtype == "train":
			if len(data_neg) < len(data_pos): data_pos = data_pos[:len(data_neg)]
			else: data_neg = data_neg[:len(data_pos)]

		label_pos = [1] * len(data_pos)
		label_neg = [0] * len(data_neg)

		self.data = data_pos + data_neg
		self.labels = label_pos + label_neg
		self.corpus = corpus
		self.sent_size = sent_size

	def tokenize(self, sent):
		word_tokens = nltk.word_tokenize(sent)
		tokens = [w.lower() for w in word_tokens if w.isalnum()]
		return tokens

	def __getitem__(self, index):
		encoded_words = torch.LongTensor(np.zeros(self.sent_size, dtype=np.int64))
		count = 0
		tokens = self.tokenize(self.data[index])
		for token in tokens:
			if token in self.corpus.dictionary.words:
				if count > self.sent_size - 1: break
				encoded_words[count] = self.corpus.dictionary.word2idx[token]
				count += 1
		label = torch.LongTensor([self.labels[index]])
		return encoded_words, label

	def __len__(self): return len(self.data) 