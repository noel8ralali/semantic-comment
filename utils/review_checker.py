import torch
import numpy as np
import pickle
import nltk
from torch.autograd import Variable

model = torch.load("data/model/lstm_model.pt")
corpus_file = open("data/dicts/corpus.pickle", "rb")
corpus = pickle.load(corpus_file)
sent_size = 10
#model.batch_size = 1

def tokenize(sent):
	word_tokens = nltk.word_tokenize(sent)
	tokens = [w.lower().strip() for w in word_tokens if w.isalnum()]
	return tokens

def encode_words(sent):
	encoded_words = torch.LongTensor(np.zeros(sent_size, dtype=np.int64))
	count = 0
	tokens = tokenize(sent)
	for token in tokens:
		if count == sent_size - 1: break
		if token in corpus.dictionary.words:
			encoded_words[count] = corpus.dictionary.word2idx[token]
			count+=1
	return encoded_words

def check(sent):
	encoded_words = encode_words(sent)
	data = Variable(encoded_words)
	model.batch_size = 1
	model.hidden = model.init_hidden()
	output = model(data)
	_, pred = torch.max(output.data, 1)
	if pred.item() == 1 : print("positive")
	else:  print("negative")
	

