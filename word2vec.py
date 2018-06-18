import gensim
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

os.chdir("../Cleaned Data")
sentences = LineSentence("conversationData.txt")
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
print('Saving the word embedding matrix')
fname = "embeddingMatrix.npy"

model.wv.save_word2vec_format(fname, 'numpy.float32')

