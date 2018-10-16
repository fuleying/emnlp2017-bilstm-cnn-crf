#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
import nltk
import os
import sys
from neuralnets.ConStringBiLSTM import ConStringBiLSTM
from util.preprocessing import addCasingInformation, addCharInformation
from util.preprocessing import addEmbeddings, createMatrices, readCoNLL
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings

charlm_embedding_forward = CharLMEmbeddings('news-forward')
charlm_embedding_backward = CharLMEmbeddings('news-backward')
# create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(embeddings=[charlm_embedding_forward,
                                                   charlm_embedding_backward])


pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
os.chdir(pardir)

if len(sys.argv) < 3:
    print("Usage: python RunModel_CoNLL_Format.py modelPath inputPathToConllFile")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]
inputColumns = {0: "tokens"}

# Load the model
lstmModel = ConStringBiLSTM.loadCRFModel(modelPath)


# Prepare the input
sentences = readCoNLL(inputPath, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)

# Map casing and character information to integer indices
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# Perform the word embedding / Contextual String embedding lookup
embLookup = lstmModel.embeddingsLookup
if (embLookup.ConStringEmbedModel == None):
    embLookup.ConStringEmbedModel = stacked_embeddings
addEmbeddings(dataMatrix, embLookup.sentenceLookup)


# Tag the input
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
