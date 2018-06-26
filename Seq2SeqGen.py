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
os.chdir("../Cleaned Data")

def createTrainingMatrices(conversationFileName, maxLen):
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
				encoderMessage[keyIndex] = 0
				# Check if this part works correctly? If it doesn't, append eos to origi wordlist
		encoderMessage[keyIndex + 1] = wordVecFloatToInt(float(min(model.wv.vectors)-1))
		# Integerize the decoder string
		for valueIndex, word in enumerate(valueSplit):
			try:
				decoderMessage[valueIndex] = wordVecFloatToInt(float(model.wv[word]))
			except:
				# print(word)
				decoderMessage[valueIndex] = 0
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
	print(xTrain.shape)
	print(yTrain.shape)
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


def getTestInput(inputMessage, maxLen):
	global model
	encoderMessage = np.full((maxLen), wordVecFloatToInt(float(
		max(model.wv.vectors)+1)), dtype='int32')
	inputSplit = inputMessage.lower().split()
	for index,word in enumerate(inputSplit):
		try:
			encoderMessage[index] = wordVecFloatToInt((float(model.wv[word])))
		except ValueError:
			continue
	encoderMessage[index + 1] = wordVecFloatToInt(float(min(model.wv.vectors)-1))
	encoderMessage = encoderMessage[::-1]
	encoderMessageList=[]
	for num in encoderMessage:
		encoderMessageList.append([num])
	return encoderMessageList

def idsToSentence(ids):
	global model
	global wordVecsInts
	global wordsAndVecs
	EOStokenIndex = wordVecFloatToInt(float(min(model.wv.vectors)-1))
	padTokenIndex = wordVecFloatToInt(float(max(model.wv.vectors)+1))
	myStr = ""
	listOfResponses=[]
	for num in ids:
		if (num[0] == EOStokenIndex or num[0] == padTokenIndex):
			listOfResponses.append(myStr)
			myStr = ""
		else:
			myStr = myStr + wordsAndVecs[(list(wordVecsInts.keys())
			                 [list(wordVecsInts.values()).index(num[0])])] + " "
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
# print(count)

all_words = list(model.wv.vocab)
wordsAndVecs = {}
for word in all_words:
	wordsAndVecs[float(model.wv[word])] = word
	wordsAndVecs[float(min(model.wv.vectors)-1)] = '<EOS>'
	wordsAndVecs[float(max(model.wv.vectors)+1)] = '<pad>'
	wordsAndVecs[0] = ' '

def wordVecFloatToInt(wordfloat):
	global model
	global wordVecsInts
	return wordVecsInts[wordfloat]


if (os.path.isfile('Seq2SeqXTrain.npy') and os.path.isfile('Seq2SeqYTrain.npy')):
	xTrain = np.load('Seq2SeqXTrain.npy')
	yTrain = np.load('Seq2SeqYTrain.npy')
	print('Finished loading training matrices')
	numTrainingExamples = xTrain.shape[0]
else:
	numTrainingExamples, xTrain, yTrain = createTrainingMatrices(
	    'conversationDictionary.npy', maxEncoderLength)
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
															count, count, embeddingDim, feed_previous=feedPrevious)

decoderPrediction = tf.argmax(decoderOutputs, 2)

lossWeights = [tf.ones_like(l, dtype=tf.float32) for l in decoderLabels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, decoderLabels, lossWeights, count)
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
		print(('Current loss: ', curLoss, 'at iteration', i))
		summary = sess.run(merged, feed_dict=feedDict)
		writer.add_summary(summary, i)
	if (i % 25 == 0 and i != 0):
		num = randint(0,len(encoderTestStrings) - 1)
		print(encoderTestStrings[num])
		inputVector = getTestInput(encoderTestStrings[num], maxEncoderLength);
		feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
		feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
		feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
		feedDict.update({feedPrevious: True})
		ids = (sess.run(decoderPrediction, feed_dict=feedDict))
		print(idsToSentence(ids))
		

	if (i % 10000 == 0 and i != 0):
		savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)
