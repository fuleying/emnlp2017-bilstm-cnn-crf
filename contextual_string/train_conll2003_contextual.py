from __future__ import print_function
import os
import logging
import sys
from neuralnets.ConStringBiLSTM import ConStringBiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from neuralnets.ContextualStringEmbeddings import ContextualStringEmbeddings
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=4,
                                                   inter_op_parallelism_threads=4)))

##################################################

# Change into the parent dir of the script
pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
os.chdir(pardir)


# Logging level
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
    'conll2003':                                   #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 3:'ner_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 2 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'ner_BIO',                                #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': '-DOCSTART'}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}

# Prepares the dataset to be used with the LSTM-network.
# Creates and stores cPickle files in the pkl/ folder
embeddings_file = 'data/Lambert/embeddings/glove.6B.100d.txt'

embLookup = ContextualStringEmbeddings(embeddings_file)
# You can use a cache to precompute the Contextual String Embeddings once.
# See Create_ConString_Cache.py for an example.
embLookup.loadCache('data/Lambert/embeddings/constring_cache_conll2003.pkl')

pickleFile = perpareDataset(datasets, embLookup)


######################################################
#
# The training of the network starts here
#
######################################################

#Load the embeddings and the dataset
mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
# params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25)}

# # BiLSTM-CNNs-CRF Model
# params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25),
#           'charEmbeddings': 'CNN', 'maxCharLength': 50}

# BiLSTM-LSTM-CRF Model
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25),
          'charEmbeddings': 'LSTM', 'maxCharLength': 50}


model = ConStringBiLSTM(embLookup, params)
model.setMappings(mappings)
model.setDataset(datasets, data)
model.storeResults('data/Lambert/results/conll2003_constring.csv')
model.modelSavePath = "data/Lambert/models/constring_[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=50)
