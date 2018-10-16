from __future__ import print_function
import os
import logging
import sys
from neuralnets.ELMoBiLSTM import ELMoBiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings
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
    'conll2000_chunking':                                   #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'chunk_BIO',                              #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}


"""
# https://allennlp.org/elmo  Original (5.5B)
# elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
elmo_options_file = {
    "lstm": {"use_skip_connections": true,
             "projection_dim": 512,
             "cell_clip": 3,
             "proj_clip": 3,
             "dim": 4096,
             "n_layers": 2},
    "char_cnn": {"activation": "relu",
                 "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256],
                             [6, 512], [7, 1024]],
                 "n_highway": 2,
                 "embedding": {"dim": 16},
                 "n_characters": 262,
                 "max_characters_per_token": 50}}
"""


# Prepares the dataset to be used with the LSTM-network.
# Creates and stores cPickle files in the pkl/ folder
# elmo_options_file= 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
# elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
elmo_options_file = 'data/Lambert/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo_weight_file = 'data/Lambert/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
embeddings_file = 'data/Lambert/embeddings/komninos_english_embeddings.gz'


elmo_mode = 'weighted_average'
elmo_cuda_device = -1 #Which GPU to use. -1 for CPU

embLookup = ELMoWordEmbeddings(embeddings_file, elmo_options_file,
                               elmo_weight_file, elmo_mode, elmo_cuda_device)
# You can use a cache to precompute the ELMo embeddings once.
# See Create_ELMo_Cache.py for an example.
embLookup.loadCache('data/Lambert/embeddings/elmo_cache_conll2000_chunking.pkl')

pickleFile = perpareDataset(datasets, embLookup)


######################################################
#
# The training of the network starts here
#
######################################################

#Load the embeddings and the dataset
mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100,100], 'dropout': (0.5, 0.5)}

model = ELMoBiLSTM(embLookup, params)
model.setMappings(mappings)
model.setDataset(datasets, data)
model.modelSavePath = "data/Lambert/models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=25)
