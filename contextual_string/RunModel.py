#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
import nltk
import os
import sys
from util.preprocessing import addCharInformation, createMatrices
from util.preprocessing import addCasingInformation, addEmbeddings
from neuralnets.ConStringBiLSTM import ConStringBiLSTM
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings

charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')
# create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(embeddings=[charlm_embedding_forward,
                                                   charlm_embedding_backward])


pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
os.chdir(pardir)

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

# Load the model
lstmModel = ConStringBiLSTM.loadCRFModel(modelPath)


# Read input
with open(inputPath, 'r') as f:
    text = f.read()

# Prepare the input
sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
addCharInformation(sentences)
addCasingInformation(sentences)

# Map casing and character information to integer indices
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# Get the embedding lookup class
embLookup = lstmModel.embeddingsLookup
if (embLookup.ConStringEmbedModel == None):
    embLookup.ConStringEmbedModel = stacked_embeddings

# Perform the word embedding / Contextual String embedding lookup
addEmbeddings(dataMatrix, embLookup.sentenceLookup)


# Use the model to tag the input
tags = lstmModel.tagSentences(dataMatrix)

# Output to stdout
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']

    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

        print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    print("")
