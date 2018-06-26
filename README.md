## Seq2SeqBot
This repo contains code for using whatsapp chats to train a simple Bot using the seq2seq algorithm in Tensorflow. 
Pre-processing scripts for whatsapp chats is provided. The chats are however not provided (For Privacy purposes)

## Folders
After cloning this repo in a new folder, create two folders names "Data" and "Cleaned Data". 
Export your individual Whatsapp chats to a .txt file and place them in the folder called "Data".

## Runtime
The focus is on the files "chat_cleaner.py" and "Seq2SeqGen".
"chat_cleaner.py" prepares your whatsapp chats by removing names, dates, emojis and other unnnecessary data.
"Seq2SeqGen.py" does the actual training of your customized bot. 
Remember to change the _test strings_ that we'll use as input at intervals during training to give you a feel of
how fast/well your bot is learning.

## Future Work
Please help make our chatbot better by contributing to this repo.
I'm working on having the input vectors of the LSTM to be of larger dimensions i.e. each word to be converted to a vector of say 
100 dimensions using Gensim's Word2Vec.
This is work in progress :)


