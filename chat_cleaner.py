import os
import emoji 
import re
import numpy as np
import pickle



def give_emoji_free_text(text, my_file_string, continuity):
    # allchars = [str for str in text.decode('utf-8')]
	allchars = text
	emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
	clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
	# Deal with some weird tokens
	clean_text = clean_text.replace("\xc2\xa0", "")
	# Remove links from chats
	clean_text = re.sub(r'http\S+', '', clean_text)
	# Remove punctuation
	clean_text = re.sub('([.,!?*_()`~/<>=])', ' ', clean_text)
	# Remove multiple spaces in message
	clean_text = re.sub(' +', ' ', clean_text)
	if len(clean_text) <=3:
		return my_file_string
	else:		
		clean_text = clean_text + "\n"
		if continuity:
			clean_text = " " + clean_text
		my_file_string = my_file_string + clean_text
		return my_file_string

def messageDictionarymaker():
	os.chdir("../Cleaned Data")
	convo_count = 0
	messageDictionary = {}
	conversationFile = open('conversationTxtDicts.txt', 'a', encoding='utf-8')
	print("Combining the cleaned chats into a single dictionary: ")
	for fname in os.listdir():
		if fname[-3:] == "txt" and fname[:7] == "cleaned":
			print(fname)
			anonymous_file = open(fname, 'r', encoding='utf-8')
			otherPersonsMessage, myMessage = "", ""
			previous_name = ""
			file_not_end = True
			while file_not_end:
				where = anonymous_file.tell()
				line = anonymous_file.readline()
				if not line:
					file_not_end = False
				else:
					try:
						name = line.split(":")[0]
						part_content = line.split(":")[1]
						part_content = part_content.replace('\n', ' ').lower()
						if (previous_name != name) and myMessage and otherPersonsMessage: #Determines how many messages will make a dialogue
						# if myMessage and otherPersonsMessage:
									messageDictionary[otherPersonsMessage.rstrip()] = myMessage.rstrip()
									otherPersonsMessage, myMessage = "", ""
									convo_count += 1
						else:
							if not (previous_name == name):
								if name == "B":
									otherPersonsMessage = otherPersonsMessage + part_content + " "
									# otherPersonsMessage = part_content + " "
								else:
									myMessage = myMessage + part_content + " "
									# myMessage = part_content + " "
						previous_name = name
					except:
						pass
			anonymous_file.close()
	np.save('conversationDictionary.npy', messageDictionary)
	# for key, value in messageDictionary.items():
	# 	if (not key.strip() or not value.strip()):
	# 		# If there are empty strings
	# 		continue
	# 	conversationFile.write(key.strip() + value.strip())
	print("Saved combined numpy dictionaries!")
	print(convo_count)
	conversationFile.write(str(messageDictionary))
	conversationFile.close()
	
def cleaner():
	os.chdir("../Data")
	for fname in os.listdir():
		if fname[-3:] =="txt":
			cfname = "../Cleaned Data/" + "cleaned_" + fname
			dirty_file = open(fname, 'r', encoding='utf-8')
			clean_file = open(cfname, 'a', encoding='utf-8')
			my_file_string = ""
			# active_pers = " "
			names = {}
			first_line = True
			replacement = 65
			file_not_end = True
			while file_not_end:
				where = dirty_file.tell()
				line = dirty_file.readline()
				if not line:
					# dirty_file.seek(where)
					file_not_end = False
				else:
					if first_line:
						first_line = False
					else:
						# print(line)
						date_section = line[0:10]
						# print(date_section)
						pay_load = line[20:]
						try:
							name = pay_load.split(":")[0]
							part_content = pay_load.split(":")[1]
							# print(part_content)
							
							if part_content == " <Media omitted>\n":
								# print("Ola!")
								pass
							
							else:
								current_name = name
								if name in names.keys():
									pass
									# print(name)
									# pay_load = pay_load.replace(name, names[name], 1)

								else:
									# current_name = name
									names[str(name)] = chr(replacement)
									# print(names)
									# pay_load = pay_load.replace(name, names[name], 1)
									replacement += 1
									# print(pay_load)
								pay_load = pay_load.replace(name, names[name], 1)
								to_write = str(pay_load)
								# clean_file.write(to_write)
								my_file_string = give_emoji_free_text(to_write, my_file_string, 0)
						except Exception as e:
							# clean_file.close()
							pay_load = line
							my_file_string = my_file_string[:-1]
							to_write = str(pay_load)
							my_file_string = give_emoji_free_text(to_write, my_file_string, 1)
							# print(pay_load)
							# print(e)
							# pass				
					# prev_pers = active_pers
			# print(names)
			clean_file.write(my_file_string)
			clean_file.close()
			dirty_file.close()
			print("Finished cleaning file ", fname, " and saved it as ", cfname)

		else:
			pass

def wordListGen():
	words = []
	conversationFile = open('conversationData.txt', 'a', encoding='utf-8')
	print("Compiling all chats into one Conversation file...")
	print("Generating word list ...")
	for fname in os.listdir():
		if fname[-3:] == "txt" and fname[:7] == "cleaned":
			anonymous_file = open(fname, 'r', encoding='utf-8')
			file_not_end = True
			while file_not_end:
				where = anonymous_file.tell()
				line = anonymous_file.readline()
				if not line:
					file_not_end = False
				else:
					try:
						# Remove punctuation
						line = re.sub('([.,!?])', '', line)
						line = line[3:]
						conversationFile.write(line)
						# line = line.replace('\n', ' ').lower()
						# for word in line.split(" "):
						# 	words.append(word)
					except:
						pass
	# uniqueWords = list(set(words))
	# wordsFile = open('wordList.txt', 'w', encoding='utf-8')
	# wordsFile.write(uniqueWords)
	# wordsFile.close()
	
	# with open("wordList.txt", "wb") as fp:
	# 	pickle.dump(uniqueWords, fp)
	# fp.close()
	conversationFile.close()


if __name__ == '__main__':
	cleaner()
	messageDictionarymaker()
	wordListGen()
