import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,Bidirectional,Embedding
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
## Plotly
import plotly.offline as py
import plotly.graph_objs as go
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
# Others
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
import itertools
import random
from sklearn.manifold import TSNE
import string
import sklearn
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import operator
from afinn import Afinn
from sklearn.metrics import confusion_matrix

words = set(nltk.corpus.words.words())
words = set([i.lower() for i in words])
stops = set(stopwords.words("english"))
df = pd.read_csv('tweet_data.csv')

# convert label into binary representation 0 is negative 1 is positive 
# df = df[df['label']!= 'neutral']
df['label_binary'] =0
df.loc[df['label']=='neutral','label_binary'] =1
df.loc[df['label']=='positive','label_binary'] =2

tokenized_sents = [TweetTokenizer().tokenize(i) for i in df.tweet_text.tolist()]
for i in range(len(tokenized_sents)):
#    tokenized_sents[i] = [w for w in tokenized_sents[i] \
#                   if  w.lower() in words]
    tokenized_sents[i] = [t for t in tokenized_sents[i] if t not in string.punctuation]
    tokenized_sents[i] = [t.lower() for t in tokenized_sents[i]]
    tokenized_sents[i] = [w for w in tokenized_sents[i] if not w in stops]
    tokenized_sents[i] = [w for w in tokenized_sents[i] if w[0:3]!='htt']
    tokenized_sents[i] = [w for w in tokenized_sents[i] if w[0]!='@']

combined_word = [ " ".join(i) for i in tokenized_sents]

max_words = 2000
max_len = 30
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(combined_word)
sequences = tok.texts_to_sequences(combined_word)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
X = sequences_matrix.copy()

Y = np.zeros((len(tokenized_sents), len(np.unique(df.label_binary))), dtype=np.uint8)
for i in range(len(tokenized_sents)):
    Y[i][df.label_binary.tolist()[i]]=1

# test training split
random.seed(3)
rand_index = random.sample(range(len(Y)), len(Y))
X_shuffle = X[rand_index]
Y_shuffle = Y[rand_index]
split_index = int(len(Y)*0.8)
X_train = X_shuffle[:split_index,:]
Y_train = Y_shuffle[:split_index,]
X_test = X_shuffle[split_index:,:]
Y_test = Y_shuffle[split_index:,]

# nn model CNN and RNN, this part can be changed to test different model config
model = Sequential()
model.add(Embedding(2000, 64, input_length=30))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(256,activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer=RMSprop(),loss = 'categorical_crossentropy',metrics=['accuracy'])

train_history = model.fit(X_train,Y_train,batch_size=32,epochs=10,
          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)],
          validation_data=[X_test, Y_test])


# example with RNN only
#model = Sequential()
#model.add(Embedding(2000, 64, input_length=30))
#model.add(Bidirectional(LSTM(64)))
#model.add(Dense(256,activation='relu'))
#model.add(Dense(3, activation='sigmoid'))
#model.compile(optimizer=RMSprop(),loss = 'categorical_crossentropy',metrics=['accuracy'])
#
#train_history = model.fit(X_train,Y_train,batch_size=32,epochs=10,
#          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)],
#          validation_data=[X_test, Y_test])

y_pred = model.predict(X_test)
idx = np.argmax(y_pred, axis=-1)
pred_result = np.zeros(y_pred.shape )
pred_result[np.arange(pred_result.shape[0]), idx] = 1
temp = [list(i).index(1) for i in list(pred_result)]

Y_test1 = [list(i).index(1) for i in list(Y_test)]
confusion_matrix(Y_test1,temp)

# Use afinn as comparsion
afinn = Afinn()
score = []
for i in range(len(df.index)):
    score.append(afinn.score(df.iloc[[i]]['tweet_text'].tolist()[0]))

df['score'] = score
df['pred'] = 0
df.loc[df['score']<0,'pred'] = 0
df.loc[df['score']==0,'pred'] = 1
df.loc[df['score']>0,'pred'] = 2

len(df[df['pred']==df['label_binary']])

# confusion matrix with afinn
len(df)
confusion_matrix(df['pred'], df['label_binary'])


df111= pd.DataFrame()
df111['aaa'] = Y_test1
len(df111[df111['aaa']==1])
len(df111)

# vader
analyser = SentimentIntensityAnalyzer()
score = []
for i in range(len(df.index)):
    my_dict = analyser.polarity_scores((df.iloc[[i]]['tweet_text'].tolist()[0]))
    if 'compound' in my_dict: del my_dict['compound']
    score.append(max(my_dict.items(), key=operator.itemgetter(1))[0])
    print(i)

df['score'] = score
df['pred'] = 0
df.loc[df['score']=='neg','pred'] = 0
df.loc[df['score']=='neu','pred'] = 1
df.loc[df['score']=='pos','pred'] = 2
len(df[df['pred']==df['label_binary']])


confusion_matrix(df['pred'], df['label_binary'])
