import matplotlib.pyplot as plt
import numpy as np

def plot_acc_loss(acc, loss):

	fig = plt.figure(1)
	skip = int(len(acc) / 100)
	loss = [loss[x] for x in range(0, len(loss), skip)]
	acc = [acc[x] for x in range(0, len(acc), skip)]
	ax1 = fig.add_subplot(111)
	ln1 = ax1.plot(loss, 'r', label='Loss')
	ax2 = ax1.twinx()
	ln2 = ax2.plot(acc, 'b', label='Accuracy')

	ln = ln1 + ln2
	labs = [l.get_label() for l in ln]
	ax1.legend(ln, labs, loc=7)

	ax1.set_ylabel('Loss')
	ax1.set_title("Training loss and accuracy")
	ax1.set_xlabel('Epoch')
	ax2.set_ylabel('Accuracy')

	plt.show()

def plot_metric(tp, fp, fn, tn):
	labels = ('True pos', 'False pos', 'False neg', 'True neg')
	y_pos = np.arange(len(labels))
	scores = [tp, fp, fn, tn]
	plt.bar(y_pos, scores, align='center', alpha=0.5)
	plt.xticks(y_pos, labels)
	plt.title('Predictive Value')
	plt.show()

	labels = ('Accuracy', 'Precission', 'Recall', 'F1 Score')
	y_pos = np.arange(len(labels))
	acc = (tp + tn) / (tp + fp + fn + tn)
	precission = tp / (tp + fp)
	recall = tp / (tp + fn) 
	f1_score = 2 * (precission * recall) / (precission + recall)
	scores = [acc, precission, recall, f1_score]
	plt.bar(y_pos, scores, align='center', alpha=0.5)
	plt.xticks(y_pos, labels)
	plt.title('Evaluation Metrics')
	plt.show()
