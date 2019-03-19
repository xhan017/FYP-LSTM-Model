import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Input data files are available in the "./input_data/" directory.
import os
print(os.listdir("../input_data/final"))
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
import matplotlib.pyplot as plt
# # %matplotlib inline
from pandas.plotting import table

import time
# import json
# from argparse import ArgumentParser

###################################
## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# ============== Set and Tune ============== 
# ========================================== 

# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
# split positives dataset into train, val, and test
proportion_train = 0.64 #0.7 
proportion_val = 0.16 #0.1 
proportion_test = 0.2 #0.20

# set number of samples in each set
trainSize = 550000 #350000 #30000 #11000

# Init
epochs = 80
emb_dim = 128 #150
batch_size = 256 #512   

# LSTM Model
# tune Dropout(0.6) and LSTM 64 or 32
ratioFP = 0.5
FPtrain_frac = 0.8
dropout = 0.5
LSTM_units = 64

# Model name:
modelName = 'V5-FYP-v2'

# ================================================#
# ============== Defining functions ==============#
# ================================================#
# Setting up tokenizer
n_most_common_words = 1000 #8000
max_len = 150
tokenizer = Tokenizer(num_words=n_most_common_words, lower=False)

# This function should only be called once
def initializeTokenizer(trainingDF):
    tokenizer.fit_on_texts(trainingDF['OPCODE'].values)
    # Saving tokenizer
    import pickle
    with open('./PickleJar/'+modelName+'_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def label(df):    
    # label data
    df['LABEL'] = 0
    df.loc[df['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
    df.loc[df['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1
    
def preprocess(df):vb
    #n_most_common_words = 1000 #8000
    #max_len = 130

    # Class Tokenizer - This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
    # tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    #tokenizer = Tokenizer(num_words=n_most_common_words, lower=False)

    # fit_on_texts - Updates internal vocabulary based on a list of texts. In the case where texts contains lists, we assume each entry of the lists to be a token.
    # tokenizer.fit_on_texts(increased_vul['OPCODE'].values)
    #tokenizer.fit_on_texts(df['OPCODE'].values)

    # # Transforms each text in texts in a sequence of integers.
    sequences = tokenizer.texts_to_sequences(df['OPCODE'].values)
    # sequences = tokenizer.texts_to_sequences(tt)

    #Find number of unique words/tokens
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    #pad sequences with zeros in front to make them all maxlen
    X = pad_sequences(sequences, maxlen=max_len)
    return X

def dftoXY(df):
    # Save test X and y
    X_test = preprocess(df)
    # label data
    label(df)
    print(pd.value_counts(df['LABEL']))
    y_test = to_categorical(df['LABEL'], num_classes=2)
    return X_test, y_test    

def XandY(posdf, negdf):
    dfset = pd.concat([posdf, negdf])
    dfset = dfset.sample(frac=1, random_state=39, replace=False)

    dfset['LABEL'] = 0

    #One-hot encode the lab
    dfset.loc[dfset['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
    dfset.loc[dfset['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1
    # df_train.head()

    X, y = dftoXY(dfset)

    print('Shape of X: {}'.format(X.shape))

    # for sm.fit_sample
    y_labels = np.expand_dims(np.array(np.argmax(y, axis=1)), axis=1)
    print('Shape of y: {}'.format(y_labels.shape))

    return X, y_labels


# ===============================================================#
# ============== Loading and reading csv input data =============# 
# ===============================================================#
dataset = 'clean_train.csv'
data = pd.read_csv('../input_data/final/'+dataset, usecols=['ADDRESS', 'OPCODE', 'CATEGORY'])

shuffled = data

n = shuffled[shuffled['CATEGORY'] == '1 0 0 0'] # no vulnerabilities
s = shuffled[shuffled['CATEGORY'] == '0 1 0 0'] # suicidal
p = shuffled[shuffled['CATEGORY'] == '0 0 1 0'] # prodigal
g = shuffled[shuffled['CATEGORY'] == '0 0 0 1'] # greedy
sp = shuffled[shuffled['CATEGORY'] == '0 1 1 0'] # suicidal and prodigal

concated = pd.concat([n,s,p,g,sp], ignore_index=True)

#Shuffle the dataset
concated = concated.reindex(np.random.permutation(concated.index))
concated['LABEL'] = 0

#One-hot encode the lab
concated.loc[concated['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
concated.loc[concated['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1


# ========== set of vul contracts ========== 
# shuffle positives dataset
positives = pd.concat([s,p,g,sp])
positives_shuf = positives.sample(frac=1, random_state=39, replace=False)

num_pos_train = round(len(positives_shuf) * proportion_train)
num_pos_val = round(len(positives_shuf) * proportion_val)

pos_train = positives_shuf.iloc[0:num_pos_train] 
pos_val = positives_shuf.iloc[num_pos_train:(num_pos_train+num_pos_val)]
pos_test = positives_shuf.iloc[(num_pos_train+num_pos_val):]

print("Total number of vulnerable contracts (positives_shuf): "+str(len(positives_shuf)))
print("Train \t\tpos_train: \t\t"+str(len(pos_train)))
print("Validation \tpos_val: \t\t"+str(len(pos_val)))
print("Test \t\tpos_test: \t\t"+str(len(pos_test)))


# ========== set of non-vul contracts ========== 
# # shuffle set n
n_shuf = n.sample(frac=1, random_state=39, replace=False)

# # set number of samples in each set
num_neg_train = 550000
print(num_neg_train)
# set number of samples in each set
num_neg_train = round(num_neg_train*0.5) #30000 #11000
num_neg_val = round(((trainSize)/proportion_train)*proportion_val)
num_neg_test = round(((trainSize)/proportion_train)*proportion_test)

neg_train = n_shuf.iloc[0:num_neg_train]
neg_val = n_shuf.iloc[num_neg_train:(num_neg_train+num_neg_val)]
neg_test = n_shuf.iloc[(num_neg_train+num_neg_val):(num_neg_train+num_neg_val+num_neg_test)]
neg_notused = n_shuf.iloc[(num_neg_train+num_neg_val+num_neg_test):]

print("Number of negative samples not used: ", len(neg_notused))
print("Total number of non-vulnerable contracts (n_shuf): "+str(len(n_shuf)))
print("Train \t\tneg_train: \t\t"+str(len(neg_train)))
print("Validation \tneg_val: \t\t"+str(len(neg_val)))
print("Test \t\tneg_test: \t\t"+str(len(neg_test)))

### ============ Resampling samples ============ ###
# Prepare train set 
#X_train, ytrain_labels = XandY(pos_train, neg_train)
# Join training set together and shuffle
df_train = pd.concat([pos_train, neg_train])
df_train = df_train.sample(frac=1, random_state=39, replace=False)

# One-hot encode the lab
df_train['LABEL'] = 0
df_train.loc[df_train['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
df_train.loc[df_train['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1
# df_train.head()

# Initialize tokenizer with training set 
initializeTokenizer(df_train)

# Get train & target vectors 
X_train, y_train = dftoXY(df_train)
print('Shape of X: {}'.format(X_train.shape))

# for sm.fit_sample
ytrain_labels = np.expand_dims(np.array(np.argmax(y_train, axis=1)), axis=1)
print('Shape of y: {}'.format(ytrain_labels.shape))

## Prepare validation set 
#X_val, yval_labels = XandY(pos_val, neg_val)
df_val = pd.concat([pos_val, neg_val])
df_val = df_val.sample(frac=1, random_state=39, replace=False)

df_val['LABEL'] = 0

#One-hot encode the lab
df_val.loc[df_val['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
df_val.loc[df_val['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1
# df_train.head()

X_val, y_val = dftoXY(df_val)

print('Shape of X: {}'.format(X_val.shape))

# for sm.fit_sample
yval_labels = np.expand_dims(np.array(np.argmax(y_val, axis=1)), axis=1)
print('Shape of y: {}'.format(yval_labels.shape))

# Prepare test set 
#X_test, ytest_labels = XandY(pos_test, neg_test)
# ============ test set ============ 
df_test = pd.concat([pos_test, neg_test])
df_test = df_test.sample(frac=1, random_state=39, replace=False)

df_test['LABEL'] = 0

#One-hot encode the lab
df_test.loc[df_test['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
df_test.loc[df_test['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1
# df_train.head()

X_test, y_test = dftoXY(df_test)

print('Shape of X: {}'.format(X_test.shape))

# for sm.fit_sample
ytest_labels = np.expand_dims(np.array(np.argmax(y_test, axis=1)), axis=1)
print('Shape of y: {}'.format(ytest_labels.shape))

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", ytrain_labels.shape)
print("Number transactions X_val dataset: ", X_val.shape)
print("Number transactions y_val dataset: ", yval_labels.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", ytest_labels.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(ytrain_labels==1)))
print("Before OverSampling, counts of label '0': {}".format(sum(ytrain_labels==0)))
print("Before OverSampling, counts of label '1': {}".format(sum(yval_labels==1)))
print("Before OverSampling, counts of label '0': {}".format(sum(yval_labels==0)))
print("Before OverSampling, counts of label '1': {}".format(sum(ytest_labels==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(ytest_labels==0)))


# ============ Resample ============ 
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=39)
X_train_res, y_train_res = sm.fit_sample(X_train, ytrain_labels.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {}'.format(y_train_res.shape))

print("After OverSampling, counts of train label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of train label '0': {}".format(sum(y_train_res==0)))


# ============== LSTM Model ==============
# # Init
# # ------------------
# # ------ tune ------
# epochs = 1 #70 #50 #100
# emb_dim = 128 #150
# batch_size = 256 #512
# # ------------------
# # ------------------

# Convert format for training  
ytrainres_cat = to_categorical(y_train_res, num_classes=2)
yvalres_cat = to_categorical(yval_labels, num_classes=2)
ytestres_cat = to_categorical(ytest_labels, num_classes=2)

#print((X_train_res.shape, ytrainres_cat.shape, X_val_res.shape, yvalres_cat.shape, X_test_res.shape, ytestres_cat.shape))
print((X_train_res.shape, ytrainres_cat.shape, X_val.shape, yvalres_cat.shape, X_test.shape, ytestres_cat.shape))

# Training  
# LSTM Model
n_most_common_words = 1000 #150
model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X_train_res.shape[1]))
model.add(SpatialDropout1D(dropout))
model.add(LSTM(LSTM_units, dropout=dropout, recurrent_dropout=dropout))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

import time
start_time = time.time()

# history = model.fit(X_train_res, ytrainres_cat, epochs=epochs, batch_size=batch_size, validation_split=0.0, validation_data=(X_val_res, yvalres_cat),callbacks=[EarlyStopping(monitor='loss',patience=7, min_delta=0.0001)])
history = model.fit(X_train_res, ytrainres_cat, epochs=epochs, batch_size=batch_size, validation_split=0.0, validation_data=(X_val, yvalres_cat),callbacks=[EarlyStopping(monitor='loss',patience=7, min_delta=0.0001)])

end_time = time.time()
print('Time taken for training: ', end_time-start_time)

# Save model to models folder 
model.save('./SavedModels/'+modelName+'.h5')   

import pickle
with open('./PickleJar/' + 'train' + modelName, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)   

# ============== Evaluation ==============

# Test accuracy
accr = model.evaluate(X_test, ytestres_cat)
print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}'.format(accr[0],accr[1]))

# To calculate precision and recall
# y_pred = model.predict_classes(X_test_res, batch_size=32, verbose=0)
y_pred = model.predict_classes(X_test, batch_size=32, verbose=0)
# ytest_true = y_test_res
ytest_true = ytest_labels

from sklearn.metrics import average_precision_score
# Compute the average precision score
average_precision = average_precision_score(ytest_true, y_pred)
print('Average Precision Score: {:0.4f}\n'.format(average_precision))

# Compute the recall
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature

precision, recall, _ = precision_recall_curve(ytest_true, y_pred)
print('Recall Score: {:0.4f}\n'.format(recall[1]))

print("============== ADDED Evaluation ==============")
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

# The set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
print('Accuracy: {:0.4f}\n'.format(accuracy_score(ytest_true, y_pred)))
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
print('Recall: {:0.4f}\n'.format(recall_score(ytest_true, y_pred)))
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
print('Precision: {:0.4f}\n'.format(precision_score(ytest_true, y_pred)))
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
print('F1 score: {:0.4f}\n'.format(f1_score(ytest_true, y_pred)))

# In binary classification, the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
print('\n confusion matrix:\n',confusion_matrix(ytest_true, y_pred))
# Overview of all scores
print('\n clasification report:\n', classification_report(ytest_true, y_pred))


# ====================================================
# ============== Save Model and Results ==============
# identifier = 'v3_150KLSTM64epoch70_train0.64'
# identifier = 'v3_testing0'


# Save Xneg_notused 
# neg_notused.to_pickle('./PickleJar/'+ 'neg_notused_' + modelName + '.pkl')

