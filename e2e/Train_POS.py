# This script trains the BiLSTM-CRF architecture for part-of-speech tagging using
# the universal dependency dataset (http://universaldependencies.org/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle


# :: Change into the parent dir of the script ::
pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
os.chdir(pardir)

# :: Logging level and formatter ::
loggingLevel = logging.INFO
formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
ch.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(loggingLevel)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'unidep_pos':                            #Name of the dataset
        {'columns': {1:'tokens', 3:'POS'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'POS',                     #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}


# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
embeddingsPath = 'data/Lambert/embeddings/komninos_english_embeddings.gz'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
# embeddings: np.array -> shape: (222313, 300)
# mappings: dict -> keys(['tokens', 'casing', 'POS', 'characters'])
# data["unidep_pos"]: dict -> keys(['trainMatrix', 'devMatrix', 'testMatrix'])
# trainMatrix: list of `sentence_dict` -> keys(['tokens', 'casing', 'POS', 'characters', 'raw_tokens'])
# 'tokens', 'casing', 'POS', 'characters' are all index list.
# 'raw_tokens' is token string list.
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25)}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('data/Lambert/results/unidep_pos_results.csv') #Path to store performance scores for dev / test
model.modelSavePath = "data/Lambert/models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Path to store models
model.fit(epochs=25)
