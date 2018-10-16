from neuralnets.ContextualStringEmbeddings import ContextualStringEmbeddings
from util.CoNLL import readCoNLL
import os
import sys
import logging
import time

if len(sys.argv) < 3:
    print("Usage: python Create_ConString_Cache.py datasetName tokenColumnId")
    exit()

# Change into the parent dir of the script
pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
os.chdir(pardir)

datasetName = sys.argv[1]
tokenColId = int(sys.argv[2])


# Logging level
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


commentSymbol = '-DOCSTART'
columns = {tokenColId: 'tokens'}


picklePath = "data/Lambert/embeddings/constring_cache_" + datasetName + ".pkl"

embLookup = ContextualStringEmbeddings(None)

print("Contextual String Embeddings Cache Generation")
print("Output file:", picklePath)

splitFiles = ['train.txt', 'dev.txt', 'test.txt']

for splitFile in splitFiles:
    inputPath = os.path.join('data/Lambert', datasetName, splitFile)

    print("Adding file to cache: " + inputPath)
    sentences = readCoNLL(inputPath, columns, commentSymbol)
    tokens = [sentence['tokens'] for sentence in sentences]

    start_time = time.time()
    embLookup.addToCache(tokens)
    end_time = time.time()
    print("%s processed in %.1f seconds" % (splitFile, end_time - start_time))
    print("\n---\n")

print("Store file at:", picklePath)
embLookup.storeCache(picklePath)
