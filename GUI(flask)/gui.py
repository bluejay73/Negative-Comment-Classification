from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from keras_tqdm import TQDMNotebookCallback

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import codecs
from tqdm import tqdm
import pandas
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, GlobalMaxPool1D, Bidirectional, GlobalMaxPooling1D
from keras.layers import LSTM, GRU, Dropout , BatchNormalization, Embedding, Flatten, GlobalAveragePooling1D, concatenate, Input
from keras.models import load_model
model=load_model('my_model.h5')
# the maximum number of words considered is 100000
MAX_NB_WORDS = 100000
# the size of the sentences will be 250
max_seq_len = 50
train = pd.read_csv('train.csv')
train.dropna(inplace=True)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_classes =list_classes[0:100000]

y_train = train[list_classes]
y_train=y_train[0:100000]
test = pd.read_csv('test.csv')
test.dropna(inplace=True)
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
raw_docs_train = train['comment_text'].tolist()
raw_docs_test = test['comment_text'].tolist()
raw_docs_train=raw_docs_train[0:100000]
raw_docs_test=raw_docs_test[0:100000]
# the maximum number of words considered is 100000
MAX_NB_WORDS = 100000
# the size of the sentences will be 250
max_seq_len = 50

raw_docs_train = train['comment_text'].tolist()
raw_docs_test = test['comment_text'].tolist()
raw_docs_train=raw_docs_train[0:100000]
raw_docs_test=raw_docs_test[0:100000]
num_classes = len(list_classes)
print raw_docs_train[0]
tokenizer = RegexpTokenizer(r'\w+')
print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
	tokens = tokenizer.tokenize(doc)
	filtered = [word for word in tokens if word not in stop_words]
	processed_docs_train.append(" ".join(filtered))

print("pre-processing test data...")
processed_docs_test = []
for doc in tqdm(raw_docs_test):
	tokens = tokenizer.tokenize(doc)
	filtered = [word for word in tokens if word not in stop_words]
	processed_docs_test.append(" ".join(filtered))


print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))

#pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

print("Done !!")



app = Flask(__name__, template_folder='template')

@app.route('/')
def my_form():
	return render_template('a.html')

@app.route('/', methods=['POST'])
def my_form_post():
	text = request.form['text']
	processed_text = text.upper()
	tokenizer = RegexpTokenizer(r'\w+')
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
	tokenizer.fit_on_texts(processed_docs_train + processed_docs_test) 
	s=processed_text
	arr=[]
	arr.append(s)
	s = tokenizer.texts_to_sequences(arr)
	s = sequence.pad_sequences(s, maxlen=max_seq_len)
	res=model.predict(s)
	final=[]
	for i in range(len(res)):
		for j in range(len(res[i])):
			final.append(str(int(res[i][j]*100))+"%")
	l=[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]
	
	ans=pandas.DataFrame(final, l)
	var1=ans.var()
	return str(ans)
if __name__ == '__main__':
	app.run()