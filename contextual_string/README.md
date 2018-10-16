# BiLSTM-CNN-CRF with ContextualStringEmbeddings-Representations for Sequence Tagging

This repository is an extension of [e2e BiLSTM-CNN-CRF implementation](http://dscoe.oocl.com/nlp/models/tree/master/e2e).

It integrates the ContextualStringEmbeddings representations from the publication [Contextual String Embeddings for Sequence Labeling](http://www.aclweb.org/anthology/C18-1139) (Alan Akbik et al., 2018) into the [e2e BiLSTM-CNN-CRF architecture](http://dscoe.oocl.com/nlp/models/tree/master/e2e) and can improve the performance significantly for different sequence tagging tasks.


The system is **easy to use**, optimized for **high performance**, and highly **configurable**.

**Requirements:**
* `Python 3.6.6` - lower versions of Python do not work
* `flair 0.2.1` - to compute the ContextualStringEmbeddings representations

```bash
pip install flair==0.2.1
```
* `Keras 2.2.0` - For the creation of BiLSTM-CNN-CRF architecture
* `Tensorflow 1.9.0` - As backend for `Keras`.


# Training
See `Train_Chunking.py` for an example how to train and evaluate this implementation. The code assumes a CoNLL formatted dataset like the CoNLL 2000 dataset for chunking.
```bash
cd contextual_string/
scp dscoehkg@STP-SHARE59-MP:~/data/models/data/Lambert/embeddings/komninos_english_embeddings.gz ../data/Lambert/embeddings/
python Train_Chunking.py
```

For training, you specify the datasets you want to train on:
```
datasets = {
    'conll2000_chunking':                                   #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'chunk_BIO',                              #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}
```

For more details, see the [e2e-bilstm-cnn-crf implementation](http://dscoe.oocl.com/nlp/models/tree/master/e2e).


# Running a stored model
If enabled during the trainings process, models are stored to the `data/Lambert/models` folder. Those models can be loaded and be used to tag new data. An example is implemented in `RunModel.py`:

```
python RunModel.py data/Lambert/models/constring_conll2000_chunking_latest.h5 data/Lambert/input.txt
```
This script will read the model `data/Lambert/models/conll2003_latest.h5` as well as the text file `data/Lambert/input.txt`. The text will be splitted into sentences and tokenized using NLTK. The tagged output will be written in a CoNLL format to standard out.

The format of `input.txt` is:
```
Rockwell International Corp.'s Tulsa unit said it signed a tentative agreement extending its contract with Boeing Co. to provide structural parts for Boeing's 747 jetliners.
He reckons the current account deficit will narrow to only 1.8 billion in September.
The group, crossing at Al Yarubiyah, Syria, was transferred to the U.N. refugee agency's camp at El Hol, 100 kilometers (60 miles) to the west, agency spokesman Peter Kessler said in Amman.
```
If your input file is in `CoNLL format` (each line a token, sentences separated by an empty line).
Please use the script `RunModel_ConLL_Format.py` as follow:

```
python RunModel_CoNLL_Format.py data/Lambert/models/constring_conll2000_chunking_latest.h5 data/Lambert/input.conll
```


# Computing ContextualStringEmbeddings representations
The computation of ContextualStringEmbeddings representations is computationally expensive. A CNN is used to map the characters of a token to a dense vectors. These dense vectors are then fed through two BiLSTMs. The representation of each token and the two outputs of the BiLSTMs are used to form the final context-dependend word embedding.

In order speed-up the training, we pre-compute the context dependend word embeddings for all sentences in our training, development, and test set. Hence, instead of passing word indizes to the BiLSTM-CRF architecture, we pass the final 1024 dimensional embeddings to the architecture.

The relevant code looks like:
```
embLookup = ContextualStringEmbeddings(embeddings_file)
pickleFile = perpareDataset(datasets, embLookup)
```

The `ContextualStringEmbeddings` provides methods for the efficient compuation of ContextualStringEmbeddings representations. It has the following parameters:
* `embeddings_file`: The Contextual String Embeddings paper concatenates traditional word embeddings, like GloVe, with the context dependent embeddings. With `embeddings_file` you can pass a path to a pre-trained word embeddings file. You can set it to `none` if you don't want to use traditional word embeddings.


The `perpareDataset` method requires the `embLookup`-object as an argument. It then iterates through all sentences in your dataset, computes the Contextual String embeddings, and stores it in a pickle-file in the `pkl/` folder.

## Pre-compute Contextual String Embeddings once
The `ContextualStringEmbeddings` class implements a caching mechansim for a quick lookup of sentences => context dependent word representations for all tokens in the sentence.

You can run `Create_ConString_Cache.py` to iterate through all you sentences in your dataset and create the Contextual String embeddings for those. It stores these embeddings in the file `data/Lambert/embeddings/constring_cache_[DatasetName].pkl`.
```python
python Create_ConString_Cache.py conll2000_chunking 0
python Create_ConString_Cache.py conll2003 0
python Create_ConString_Cache.py csnews_bio 0
```

Once you create such a cache, you can load those in your experiments:
```
embLookup = ContextualStringEmbeddings(embeddings_file)
embLookup.loadCache('data/Lambert/embeddings/constring_cache_conll2000_chunking.pkl')
pickleFile = perpareDataset(datasets, embLookup)
```

If a sentence is in the cache, the cached representations for all tokens in that sentence are used. This requires the computation of the Contextual String embeddings for a dataset must only be done once.

*Note:* The cache file can become rather large, as 3*1024 float numbers per token must be stored. The cache file requires about 3.7 GB for the CoNLL 2000 dataset on chunking with about 13.000 sentences.
