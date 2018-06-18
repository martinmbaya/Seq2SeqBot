import tensorflow as tf 
import numpy as np 
import sys
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# Removes an annoying Tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
if (os.path.isfile('embeddingMatrix.npy')):
	model = Word2Vec.load('embeddingMatrix.npy')
else:
	sentences = LineSentence("conversationData.txt")
	model = Word2Vec(sentences, size=1, window=5,
						min_count=0, iter=100, workers=4)
	print('Saving the word embedding matrix')
	fname = "embeddingMatrix.npy"
	model.save(fname)
wordVecs = []
wordVecsInts = {}
for i in list(model.wv.vectors):
    wordVecs.append(i[0])
wordVecs.append(float(min(model.wv.vectors)-1))
wordVecs.append(float(max(model.wv.vectors)+1))

wordVecs.sort()
count = 1
for i in wordVecs:
    wordVecsInts[float(i)] = count
    count += 1
def wordVecFloatToInt(wordfloat):
	global model
	global wordVecsInts
	return wordVecsInts[wordfloat]


def createTrainingMatrices(conversationFileName, wList, maxLen):
	global model
	conversationDictionary = np.load(conversationFileName).item()
	numExamples = len(conversationDictionary)
	xTrain = np.zeros((numExamples, maxLen), dtype='int32')
	yTrain = np.zeros((numExamples, maxLen), dtype='int32')
	for index,(key,value) in enumerate(conversationDictionary.items()):
		# Will store integerized representation of strings here (initialized as padding)
		encoderMessage = np.full((maxLen), wordVecFloatToInt(float(
			max(model.wv.vectors)+1)), dtype='int32')
		decoderMessage = np.full((maxLen), wordVecFloatToInt(float(
			max(model.wv.vectors)+1)), dtype='int32')
		# Getting all the individual words in the strings
		keySplit = key.split()
		valueSplit = value.split()
		keyCount = len(keySplit)
		valueCount = len(valueSplit)
		# Throw out sequences that are too long or are empty
		if (keyCount > (maxLen - 1) or valueCount > (maxLen - 1) or valueCount == 0 or keyCount == 0):
			continue
		# Integerize the encoder string
		for keyIndex, word in enumerate(keySplit):
			try:
				encoderMessage[keyIndex] = wordVecFloatToInt((float(model.wv[word])))
			except:
				# print(word)
				# TODO: This isnt really the right way to handle this scenario
				encoderMessage[keyIndex] = 0.0
				# Check if this part works correctly? If it doesn't, append eos to origi wordlist
		# encoderMessage[keyIndex + 1] = wList.index('<EOS>')
		encoderMessage[keyIndex + 1] = wordVecFloatToInt(float(min(model.wv.vectors)-1))
		# Integerize the decoder string
		for valueIndex, word in enumerate(valueSplit):
			try:
				decoderMessage[valueIndex] = wordVecFloatToInt(float(model.wv[word]))
			except:
				# print(word)
				decoderMessage[valueIndex] = 0.0
		# Check if this part works correctly? If it doesn't, append eos to origi wordlist
		decoderMessage[valueIndex + 1] = wordVecFloatToInt(float(min(model.wv.vectors)-1))
		# print(len(encoderMessage))
		xTrain[index] = encoderMessage
		yTrain[index] = decoderMessage
	# Remove rows with all zeros
	yTrain = yTrain[~np.all(yTrain == 0, axis=1)]
	xTrain = xTrain[~np.all(xTrain == 0, axis=1)]
	numExamples = xTrain.shape[0]
	# print(yTrain)
	# print(yTrain.shape)
	return numExamples, xTrain, yTrain

def getTrainingBatch(localXTrain, localYTrain, localBatchSize, maxLen):
	global model
	num = randint(0,numTrainingExamples - localBatchSize - 1)
	arr = localXTrain[num:num + localBatchSize]
	labels = localYTrain[num:num + localBatchSize]
	# Reversing the order of encoder string apparently helps as per 2014 paper
	reversedList = list(arr)
	for index,example in enumerate(reversedList):
		reversedList[index] = list(reversed(example))

	# Lagged labels are for the training input into the decoder
	laggedLabels = []
	# EOStokenIndex = wordList.index('<EOS>')
	EOStokenIndex = wordVecFloatToInt(float(min(model.wv.vectors)-1))
	padTokenIndex = wordVecFloatToInt(float(max(model.wv.vectors)+1))
	for example in labels:
		# print(example)
		eosFound = np.argwhere(example==EOStokenIndex)[0]
		shiftedExample = np.roll(example,1)
		shiftedExample[0] = EOStokenIndex
		# The EOS token was already at the end, so no need for pad
		if (eosFound != (maxLen - 1)):
			shiftedExample[eosFound+1] = padTokenIndex
		laggedLabels.append(shiftedExample)

	# Need to transpose these 
	reversedList = np.asarray(reversedList).T.tolist()
	labels = labels.T.tolist()
	laggedLabels = np.asarray(laggedLabels).T.tolist()
	# print(len(labels))
	# print(len(laggedLabels))
	return reversedList, labels, laggedLabels

def translateToSentences(inputs, wList, encoder=False):
	global model
	# EOStokenIndex = wList.index('<EOS>')
	EOStokenIndex = wordVecFloatToInt(float(min(model.wv.vectors)-1))
	padTokenIndex = wordVecFloatToInt(float(max(model.wv.vectors)+1))
	numStrings = len(inputs[0])
	numLengthOfStrings = len(inputs)
	listOfStrings = [''] * numStrings
	for mySet in inputs:
		for index,num in enumerate(mySet):
			if (num != EOStokenIndex and num != padTokenIndex):
				if (encoder):
					# Encodings are in reverse!
					listOfStrings[index] = wList[num] + " " + listOfStrings[index]
				else:
					listOfStrings[index] = listOfStrings[index] + " " + wList[num]
	listOfStrings = [string.strip() for string in listOfStrings]
	return listOfStrings

def getTestInput(inputMessage, wList, maxLen):
	global model
	encoderMessage = np.full((maxLen), wordVecFloatToInt(float(
		max(model.wv.vectors)+1)), dtype='int32')
	inputSplit = inputMessage.lower().split()
	for index,word in enumerate(inputSplit):
		try:
			encoderMessage[index] = wList.index(word)
		except ValueError:
			continue
	# encoderMessage[index + 1] = wList.index('<EOS>')
	encoderMessage[index + 1] = wordVecFloatToInt(float(min(model.wv.vectors)-1))
	encoderMessage = encoderMessage[::-1]
	encoderMessageList=[]
	for num in encoderMessage:
		encoderMessageList.append([num])
	return encoderMessageList

def idsToSentence(ids, wList):
	global model
	# EOStokenIndex = wList.index('<EOS>')
	EOStokenIndex = wordVecFloatToInt(float(min(model.wv.vectors)-1))
	padTokenIndex = wordVecFloatToInt(float(max(model.wv.vectors)+1))
	myStr = ""
	listOfResponses=[]
	for num in ids:
		if (num[0] == EOStokenIndex or num[0] == padTokenIndex):
			listOfResponses.append(myStr)
			myStr = ""
		else:
			myStr = myStr + wList[num[0]] + " "
	if myStr:
		listOfResponses.append(myStr)
	listOfResponses = [i for i in listOfResponses if i]
	return listOfResponses

# Hyperparamters
batchSize = 24
maxEncoderLength = 15
maxDecoderLength = maxEncoderLength
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3
numIterations = 500000

# Loading in all the data structures
with open("wordList.txt", "rb") as fp:
	wordList = pickle.load(fp)


vocabSize = len(wordList)

# If you've run the entirety of word2vec.py then these lines will load in 
# the embedding matrix.
# if (os.path.isfile('embeddingMatrix.npy')):
# 	wordVectors = np.load('embeddingMatrix.npy')
# 	# wordVectors = gensim.models.KeyedVectors.load_word2vec_format(
# 	# 	'embeddingMatrix.npy', 'numpy.int32', binary=True)
# 	# wordVectors = np.matrix(wordVectors)
# 	# print(type(wordVectors))
# 	print(wordVectors.shape)
# 	wordVecDimensions = wordVectors.shape[1]
# 	# print(wordVecDimensions)
# else:
# 	question = 'Since we cant find an embedding matrix, how many dimensions do you want your word vectors to be?: '
# 	wordVecDimensions = int(eval(input(question)))

# # Add two entries to the word vector matrix. One to represent padding tokens, 
# # and one to represent an end of sentence token
# padVector = np.zeros((1, wordVecDimensions), dtype='int32')
# EOSVector = np.ones((1, wordVecDimensions), dtype='int32')
# if (os.path.isfile('embeddingMatrix.npy')): 
# 	wordVectors = np.concatenate((wordVectors,padVector), axis=0)
# 	wordVectors = np.concatenate((wordVectors,EOSVector), axis=0)

# # Need to modify the word list as well
# # print(wordList[:10])
# wordList.append('<pad>')
# wordList.append('<EOS>')
# vocabSize = vocabSize + 2

if (os.path.isfile('Seq2SeqXTrain.npy') and os.path.isfile('Seq2SeqYTrain.npy')):
	xTrain = np.load('Seq2SeqXTrain.npy')
	yTrain = np.load('Seq2SeqYTrain.npy')
	print('Finished loading training matrices')
	numTrainingExamples = xTrain.shape[0]
else:
	numTrainingExamples, xTrain, yTrain = createTrainingMatrices(
	    'conversationDictionary.npy', wordList, maxEncoderLength)
	# print(xTrain)
	np.save('Seq2SeqXTrain.npy', xTrain)
	np.save('Seq2SeqYTrain.npy', yTrain)
	print('Finished creating training matrices')

tf.reset_default_graph()

# Create the placeholders
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxEncoderLength)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits, state_is_tuple=True)

#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
# Architectural choice of of whether or not to include ^

decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoderInputs, decoderInputs, encoderLSTM, 
															vocabSize, vocabSize, embeddingDim, feed_previous=feedPrevious)

decoderPrediction = tf.argmax(decoderOutputs, 2)

lossWeights = [tf.ones_like(l, dtype=tf.float32) for l in decoderLabels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, decoderLabels, lossWeights, vocabSize)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
# If you're loading in a saved model, use the following
#saver.restore(sess, tf.train.latest_checkpoint('models/'))
sess.run(tf.global_variables_initializer())

# Uploading results to Tensorboard
tf.summary.scalar('Loss', loss)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Some test strings that we'll use as input at intervals during training
encoderTestStrings = ["what's up",
					"hi",
					"hey how are you",
					"sasa",
					"where are you",
					"that girl was really cute ",
					"what is life"
					]

zeroVector = np.zeros((1), dtype='int32')

for i in range(numIterations):

	encoderTrain, decoderTargetTrain, decoderInputTrain = getTrainingBatch(xTrain, yTrain, batchSize, maxEncoderLength)
	feedDict = {encoderInputs[t]: encoderTrain[t] for t in range(maxEncoderLength)}
	feedDict.update({decoderLabels[t]: decoderTargetTrain[t] for t in range(maxDecoderLength)})
	feedDict.update({decoderInputs[t]: decoderInputTrain[t] for t in range(maxDecoderLength)})
	feedDict.update({feedPrevious: False})

	curLoss, _, pred = sess.run([loss, optimizer, decoderPrediction], feed_dict=feedDict)
	
	if (i % 50 == 0):
		print(('Current loss:', curLoss, 'at iteration', i))
		summary = sess.run(merged, feed_dict=feedDict)
		writer.add_summary(summary, i)
	if (i % 25 == 0 and i != 0):
		num = randint(0,len(encoderTestStrings) - 1)
		print(encoderTestStrings[num])
		inputVector = getTestInput(encoderTestStrings[num], wordList, maxEncoderLength);
		feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
		feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
		feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
		feedDict.update({feedPrevious: True})
		ids = (sess.run(decoderPrediction, feed_dict=feedDict))
		print(idsToSentence(ids, wordList))
		

	if (i % 10000 == 0 and i != 0):
		savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)
