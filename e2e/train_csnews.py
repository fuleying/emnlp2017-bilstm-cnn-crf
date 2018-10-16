# -*- coding=utf-8 -
"""
Created on Saturday August 21 2018

@author: Lambert
"""


from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K

# :: Change into the parent dir of the script ::
pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
os.chdir(pardir)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'csnews_bio':                            #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 3:'ner_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'ner_BIO',                 #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': '-DOCSTART'}       #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}

# Path on your computer to the word embeddings.
embeddingsPath = 'data/Lambert/embeddings/glove.6B.100d.txt'

# Prepares the dataset to be used with the LSTM-network.
# Creates and stores cPickle files in the pkl/ folder
pickleFile = perpareDataset(embeddingsPath, datasets, padOneTokenSentence=True)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters

# # No character-level embeddings
# params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25)}

# # BiLSTM-CNNs-CRF Model
# params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25),
#           'charEmbeddings': 'CNN', 'maxCharLength': 50}

# BiLSTM-LSTM-CRF Model
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25),
          'charEmbeddings': 'LSTM', 'maxCharLength': 50}


model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data, epoch_size=1656)
# Path to store performance scores for dev / test
model.storeResults('data/Lambert/results/csnews.csv')
model.modelSavePath = "data/Lambert/models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=50)
