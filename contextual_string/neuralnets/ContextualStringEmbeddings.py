import urllib.request as urllib2
import urllib.parse as urlparse
from urllib.request import urlretrieve
import logging
import numpy as np
import pickle as pkl
import os
import gzip
import sys

from flair.data import Sentence
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings


class ContextualStringEmbeddings:
    def __init__(self, embeddings_path):
        self.embeddings_path = embeddings_path
        self.embedding_name = os.path.splitext(os.path.basename(embeddings_path))[0] if embeddings_path is not None else 'None'
        self.word2Idx = None
        self.embeddings = None

        self.ConStringEmbedModel = None
        self.cache = {}
        self.lazyCacheFiles = []

    def getConfig(self):
        return {"embeddings_path": self.embeddings_path,
                "ConStringEmbedModel": self.ConStringEmbedModel}

    def sentenceLookup(self, sentences):
        # contextual string embeddings
        constring_vectors = self.getConStringEmbedding(sentences)

        # Word Embedding
        tokens_vectors = None
        if self.embeddings_path is not None:
            if self.word2Idx is None or self.embeddings is None:
                self.word2Idx, self.embeddings = self.readEmbeddings(self.embeddings_path)

            tokens_vectors = []
            for sentence in sentences:
                per_token_embedding = []
                for token in sentence['tokens']:
                    vecId = self.word2Idx['UNKNOWN_TOKEN']

                    if token in self.word2Idx:
                        vecId = self.word2Idx[token]
                    elif token.lower() in self.word2Idx:
                        vecId = self.word2Idx[token.lower()]
                    per_token_embedding.append(self.embeddings[vecId])
                per_token_embedding = np.asarray(per_token_embedding)
                tokens_vectors.append(per_token_embedding)

        out_vectors = {}
        if tokens_vectors is not None:
            out_vectors['tokens'] = tokens_vectors

        if constring_vectors is not None:
            out_vectors['constring'] = constring_vectors

        return out_vectors


    def embed_sentences(self, languageModel, sentences):
        sentence_embed_list = []
        for sentence in sentences:
            embed_list = []
            sent_string = " ".join(sentence)
            pytorch_sentence = Sentence(sent_string)
            languageModel.embed(pytorch_sentence)

            for token in pytorch_sentence:
                embed_list.append(token.get_embedding().numpy())

            sentence_embed_list.append(np.asarray(embed_list, dtype=np.float32))

        return sentence_embed_list

    def getConStringEmbedding(self, sentences):
        if len(self.lazyCacheFiles) > 0:
            self._loadLazyCache()

        constring_embeddings = []
        non_cached_sentences = []
        non_cached_sentences_indices = []

        # Lookup cached sentences
        for sentence in sentences:
            tokens = sentence['tokens']
            cache_key = tuple(tokens)
            if len(self.cache) > 0 and cache_key in self.cache:
                constring_embeddings.append(self.cache[cache_key])
            else:
                non_cached_sentences.append(tokens)
                # the index in constring_embeddings
                non_cached_sentences_indices.append(len(constring_embeddings))
                # the Placeholder in constring_embeddings
                constring_embeddings.append(None)  # len(constring_embeddings) + 1

        # Compute Contextual String Embeddings on the fly
        if len(non_cached_sentences) > 0:
            if self.ConStringEmbedModel is None:
                self.loadConStringModel()

            idx = 0
            for constring_vectors in self.embed_sentences(self.ConStringEmbedModel, non_cached_sentences):
                # constring_vectors: (sent_len, 4096)
                assert(constring_embeddings[non_cached_sentences_indices[idx]] == None)
                constring_embeddings[non_cached_sentences_indices[idx]] = constring_vectors
                idx += 1

        return constring_embeddings

    def getIdentifier(self):
        """Returns a unique identifier for this lookup function"""
        return "Contextual_String_Embeddings_" + self.embedding_name

    def loadConStringModel(self):
        # init CharLM embeddings
        charlm_embedding_forward = CharLMEmbeddings('news-forward')
        charlm_embedding_backward = CharLMEmbeddings('news-backward')

        # now create the StackedEmbedding object that combines all embeddings
        stacked_embeddings = StackedEmbeddings(embeddings=[charlm_embedding_forward,
                                                           charlm_embedding_backward])
        self.ConStringEmbedModel = stacked_embeddings

    def loadCache(self, inputPath):
        self.lazyCacheFiles.append(inputPath)

    def storeCache(self, outputPath):
        f = open(outputPath, 'wb')
        pkl.dump(self.cache, f, -1)
        f.close()

    def addToCache(self, sentences):
        if self.ConStringEmbedModel is None:
            self.loadConStringModel()

        idx = 0
        for constringEmbedding in self.embed_sentences(self.ConStringEmbedModel, sentences):
            sentence = tuple(sentences[idx])
            self.cache[sentence] = constringEmbedding

            idx += 1

    def _loadLazyCache(self):
        while len(self.lazyCacheFiles) > 0:
            inputPath = self.lazyCacheFiles.pop()

            if not os.path.isfile(inputPath):
                print("Contextual String Embeddings cache file not found:", inputPath)
                continue

            f = open(inputPath, 'rb')
            loaded_cache = pkl.load(f)
            f.close()

            if len(self.cache) == 0:
                self.cache = loaded_cache
            else:
                self.cache.update(loaded_cache)

    def readEmbeddings(self, embeddingsPath):
        filename = os.path.basename(embeddingsPath)
        if not os.path.isfile(embeddingsPath):
            if filename in ['komninos_english_embeddings.gz', 'levy_english_dependency_embeddings.gz',
                            'reimers_german_embeddings.gz']:
                self.getEmbeddings(filename, embeddingsPath)
            else:
                print("The embeddings file %s was not found" % embeddingsPath)
                exit()

        # Read in word embeddings
        logging.info("Read file: %s" % embeddingsPath)
        word2Idx = {}
        embeddings = []
        embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                                   encoding="utf8")
        embeddingsDimension = None

        for line in embeddingsIn:
            split = line.rstrip().split(" ")
            word = split[0]

            if embeddingsDimension==None:
                embeddingsDimension = len(split) - 1

            if (len(split) - 1)!=embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
                print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
                continue

            if len(word2Idx)==0:  # Add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(embeddingsDimension)
                embeddings.append(vector)

                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                # Fixed rnd seed for unknown token, so that it is always the same
                rndState = np.random.RandomState(seed=12345)
                vector = rndState.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)

                embeddings.append(vector)

            vector = np.array([float(num) for num in split[1:]])

            embeddings.append(vector)
            word2Idx[word] = len(word2Idx)

        return word2Idx, embeddings

    def getEmbeddings(self, filename, savePath):
        if not os.path.isfile(savePath):
            self.download("https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/" + filename, savePath)

    def download(self, url, savePath, silent=False):
        filename = os.path.basename(urlparse.urlparse(url).path) or 'downloaded.file'

        def get_size():
            meta = urllib2.urlopen(url).info()
            meta_func = meta.getheaders if hasattr(
                meta, 'getheaders') else meta.get_all
            meta_length = meta_func('Content-Length')
            try:
                return int(meta_length[0])
            except:
                return 0

        def kb_to_mb(kb):
            return kb / 1024.0 / 1024.0

        def callback(blocks, block_size, total_size):
            current = blocks * block_size
            percent = 100.0 * current / total_size
            line = '[{0}{1}]'.format(
                '=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
            status = '\r{0:3.0f}%{1} {2:3.1f}/{3:3.1f} MB'
            sys.stdout.write(
                status.format(
                    percent, line, kb_to_mb(current), kb_to_mb(total_size)))

        logging.info(
            'Downloading: {0} ({1:3.1f} MB)'.format(url, kb_to_mb(get_size())))
        try:
            (savePath, headers) = urlretrieve(url, savePath, None if silent else callback)
        except:
            os.remove(savePath)
            raise Exception("Can't download {0}".format(savePath))
        else:
            print()
            logging.info('Downloaded to: {0}'.format(savePath))

        return savePath
