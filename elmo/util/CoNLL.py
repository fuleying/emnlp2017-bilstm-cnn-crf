from __future__ import print_function
import os

def conllWrite(outputPath, sentences, headers):
    """
    Writes a sentences array/hashmap to a CoNLL format
    """
    if not os.path.exists(os.path.dirname(outputPath)):
        os.makedirs(os.path.dirname(outputPath))
    fOut = open(outputPath, 'w')


    for sentence in sentences:
        fOut.write("#")
        fOut.write("\t".join(headers))
        fOut.write("\n")
        for tokenIdx in range(len(sentence[headers[0]])):
            aceData = [sentence[key][tokenIdx] for key in headers]
            fOut.write("\t".join(aceData))
            fOut.write("\n")
        fOut.write("\n")


def readCoNLL(inputPath, cols, commentSymbol=None, valTransformation=None):
    """
    Reads in a CoNLL file
        cols: {1:'tokens', 3:'POS'} or {1:'tokens', 2:'NER_BIO'}
    returns:
        sentences -- list e.g. [sentence1, sentence2, sentence3...]
        for every sentence: -- dict
        e.g. {'tokens': [], 'POS': []}
        e.g. {'tokens': [], 'NER_BIO': [],
              'NER_class': [],
              'NER_IOB': [],
              'NER_IOBES': []}
    """
    sentences = []
    sentenceTemplate = {name: [] for name in cols.values()}   # {'tokens': [], 'POS': []}
    sentence = {name: [] for name in sentenceTemplate.keys()} # {'tokens': [], 'POS': []}
    hasData = False
    for line in open(inputPath):
        line = line.strip()
        # sentence end
        if len(line) == 0 or line[0]=="\n" \
                or (commentSymbol != None and line.startswith(commentSymbol)):
            if hasData:
                sentences.append(sentence)
                sentence = {name: [] for name in sentenceTemplate.keys()}
                hasData = False
            continue

        # add sentence: token, POS or NER tag
        splits = line.split()
        for colIdx, colName in cols.items():
            val = splits[colIdx]
            if valTransformation != None:
                val = valTransformation(colName, val, splits)
            sentence[colName].append(val)
        hasData = True

    # add last sentence
    if hasData:
        sentences.append(sentence)

    for name in cols.values():
        if name.endswith('_BIO'):
            iobesName = name[0:-4]+'_class'

            #Add class
            className = name[0:-4]+'_class'
            for sentence in sentences:
                sentence[className] = []
                for val in sentence[name]:
                    valClass = val[2:] if val != 'O' else 'O'
                    sentence[className].append(valClass)

            # Add IOB encoding
            """
            The CoNLL 2003 datatset for NER comes in a quite strange encoding,
            which i named IOB (not to be confused with BIO encoding).
            The B-tag (Begin-tag) is only used if two different named entities with
            the same tag are follow directly each other,
            for example (thats how it is encoded in the dataset):
                I love London => O O I-LOC
                I love London Berlin and Paris => O O I-LOC B-LOC O I-LOC
                I love London Harry Potter => O O I-LOC I-PER I-PER
            Converts the IOB encoding from CoNLL 2003 to BIO encoding can be find in:
            https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/issues/23
            """
            iobName = name[0:-4]+'_IOB'
            for sentence in sentences:
                sentence[iobName] = []
                oldVal = 'O'
                for val in sentence[name]:
                    newVal = val
                    if newVal[0] == 'B':
                        if oldVal != 'I'+newVal[1:]:
                            newVal = 'I'+newVal[1:]
                    sentence[iobName].append(newVal)
                    oldVal = newVal

            #Add IOBES encoding
            iobesName = name[0:-4]+'_IOBES'
            for sentence in sentences:
                sentence[iobesName] = []
                for pos in range(len(sentence[name])):
                    val = sentence[name][pos]
                    nextVal = sentence[name][pos+1] if (pos+1) < len(sentence[name]) else 'O'

                    newVal = val
                    if val[0] == 'B':   # B -> S
                        if nextVal[0] != 'I':
                            newVal = 'S'+val[1:]
                    elif val[0] == 'I':  # I -> E
                        if nextVal[0] != 'I':
                            newVal = 'E'+val[1:]

                    sentence[iobesName].append(newVal)

    return sentences
