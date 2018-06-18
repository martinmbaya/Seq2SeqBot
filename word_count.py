# f = open("conversationData.txt", "r", encoding="utf-8")
# line_count = 0
# word_count = 0

# for line in f:
# 	line_count += 1
# 	for word in line.split(" "):
# 		word_count += 1
# print(line_count, word_count)
# f.close()

with open("wordList.txt", "wb") as fp:
		pickle.dump(uniqueWords, fp)
	fp.close()