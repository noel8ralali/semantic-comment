import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMClassifier(nn.Module):
	def __init__(self, embeding_dim, hidden_dim, corpus_size, label_size, batch_size, use_gpu=False):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.use_gpu = use_gpu

		self.word_embeddings = nn.Embedding(corpus_size, embeding_dim)
		self.lstm = nn.LSTM(embeding_dim, hidden_dim)
		self.FC = nn.Linear(hidden_dim, label_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		if self.use_gpu:
			h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
			c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
		else:
			h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
			c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
		return (h0, c0)
	
	def forward(self, sent):
		embeds = self.word_embeddings(sent)
		x = embeds.view(len(sent), self.batch_size, -1)
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		y = self.FC(lstm_out[-1])
		return y 			