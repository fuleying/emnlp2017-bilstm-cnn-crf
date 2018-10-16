from __future__ import print_function
import logging
"""
Computes the F1 score on BIO tagged data

@author: Nils Reimers
"""


def compute_f1_token_basis(predictions, correct, O_Label):
    prec = compute_precision_token_basis(predictions, correct, O_Label)
    rec = compute_precision_token_basis(correct, predictions, O_Label)

    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1

def compute_precision_token_basis(guessed_sentences, correct_sentences, O_Label):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert(len(guessed) == len(correct))
        for idx in range(len(guessed)):

            if guessed[idx] != O_Label:
                count += 1

                if guessed[idx] == correct[idx]:
                    correctCount += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision


def compute_f1(predictions, correct, idx2Label, correctBIOErrors='No', encodingScheme='BIO'):
    """
    predictions: [num_sentences, sent_len]
    correct:     [num_sentences, sent_len]
    idx2Label: dict
    """
    label_pred = []
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])

    encodingScheme = encodingScheme.upper()

    if encodingScheme == 'IOBES':
        convertIOBEStoBIO(label_pred)
        convertIOBEStoBIO(label_correct)
    elif encodingScheme == 'IOB':
        convertIOBtoBIO(label_pred)
        convertIOBtoBIO(label_correct)

    checkBIOEncoding(label_pred, correctBIOErrors)

    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)

    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1


def convertIOBtoBIO(dataset):
    """ Convert inplace IOB encoding to BIO encoding.
    The CoNLL 2003 datatset for NER comes in a quite strange encoding,
    which i named IOB (not to be confused with BIO encoding).

    The B-tag (Begin-tag) is only used if two different named entities with
    the same tag are follow directly each other,
    for example (thats how it is encoded in the dataset):
        I love London => O O I-LOC
        I love London Berlin and Paris => O O I-LOC B-LOC O I-LOC
        I love London Harry Potter => O O I-LOC I-PER I-PER
    """
    for sentence in dataset:
        prevVal = 'O'
        for pos in range(len(sentence)):
            firstChar = sentence[pos][0]
            if firstChar == 'I':
                if prevVal == 'O' or prevVal[1:] != sentence[pos][1:]:
                    sentence[pos] = 'B'+ sentence[pos][1:] #Change to begin tag

            prevVal = sentence[pos]

def convertIOBEStoBIO(dataset):
    """ Convert inplace IOBES encoding to BIO encoding """
    for sentence in dataset:
        for pos in range(len(sentence)):
            firstChar = sentence[pos][0]
            if firstChar == 'S':
                sentence[pos] = 'B'+sentence[pos][1:]
            elif firstChar == 'E':
                sentence[pos] = 'I'+sentence[pos][1:]


def compute_precision(guessed_sentences, correct_sentences):
    """Precision = TP / (TP + FP) = TP / P'
       Recall    = TP / (TP + FN) = TP / P
    P'=(TP+FP) : A new chunk starts with 'B' in predicted sentence
    P =(TP+FN) : A new chunk starts with 'B' in corrected sentence
    TP: The chunk is predicted correctly.
        e.g. (B, BI, BII...) both in predicted and corrected sentence.
    So:
      Precision = compute_precision(guessed_sentences, correct_sentences)
      Recall    = compute_precision(correct_sentences, guessed_sentences)
    """
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]

        assert(len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B': # P'=(TP+FP) : A new chunk starts with 'B'
                count += 1
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True

                    # Scan until it no longer starts with I
                    while idx < len(guessed) and guessed[idx][0] == 'I':
                        if correctlyFound and guessed[idx] != correct[idx]:
                            correctlyFound = False
                        idx += 1

                    # The chunk in correct was longer, still 'I', not end
                    if correctlyFound and idx < len(guessed) and correct[idx][0] == 'I':
                        correctlyFound = False

                     # TP: The chunk is predicted correctly. e.g. (B, BI, BII...)
                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision

def checkBIOEncoding(predictions, correctBIOErrors):
    """Check the generated NER tag if started with 'B'.
       If started with 'I', and its prev not 'B', it's error.
    """
    errors = 0
    labels = 0

    for sentenceIdx in range(len(predictions)):
        labelStarted = False
        labelClass = None

        for labelIdx in range(len(predictions[sentenceIdx])):
            label = predictions[sentenceIdx][labelIdx]
            if label.startswith('B-'):
                labels += 1
                labelStarted = True
                labelClass = label[2:]
            elif label == 'O':
                labelStarted = False
                labelClass = None
            elif label.startswith('I-'):
                if not labelStarted or label[2:] != labelClass:
                    errors += 1
                    # just view as a new class tag started.
                    if correctBIOErrors.upper() == 'B':
                        predictions[sentenceIdx][labelIdx] = 'B-'+label[2:]
                        labelStarted = True
                        labelClass = label[2:]
                    # # just view as non-class tag.
                    elif correctBIOErrors.upper() == 'O':
                        predictions[sentenceIdx][labelIdx] = 'O'
                        labelStarted = False
                        labelClass = None
            else:
                assert(False) #Should never be reached

    if errors > 0:
        labels += errors
        logging.info("Wrong BIO-Encoding %d/%d labels, %.2f%%" % (errors, labels, errors/float(labels)*100),)


def testEncodings():
    """ Tests BIO, IOB and IOBES encoding """

    goldBIO = [['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'B-PER', 'I-PER'],
               ['O', 'B-PER', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'I-PER'],
               ['B-LOC', 'I-LOC', 'I-LOC', 'B-PER', 'B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'B-PER']]

    print("--Test IOBES--")
    goldIOBES = [['O', 'B-PER', 'E-PER', 'O', 'S-PER', 'B-PER', 'E-PER'],
                 ['O', 'S-PER', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'I-PER', 'E-PER'],
                 ['B-LOC', 'I-LOC', 'E-LOC', 'S-PER', 'B-PER', 'I-PER', 'E-PER', 'O', 'S-LOC', 'S-PER']]
    convertIOBEStoBIO(goldIOBES)

    for sentenceIdx in range(len(goldBIO)):
        for tokenIdx in range(len(goldBIO[sentenceIdx])):
            assert (goldBIO[sentenceIdx][tokenIdx]==goldIOBES[sentenceIdx][tokenIdx])

    print("--Test IOB--")
    goldIOB = [['O', 'I-PER', 'I-PER', 'O', 'I-PER', 'B-PER', 'I-PER'],
               ['O', 'I-PER', 'I-LOC', 'I-LOC', 'O', 'I-PER', 'I-PER', 'I-PER'],
               ['I-LOC', 'I-LOC', 'I-LOC', 'I-PER', 'B-PER', 'I-PER', 'I-PER', 'O', 'I-LOC', 'I-PER']]
    convertIOBtoBIO(goldIOB)

    for sentenceIdx in range(len(goldBIO)):
        for tokenIdx in range(len(goldBIO[sentenceIdx])):
            assert (goldBIO[sentenceIdx][tokenIdx]==goldIOB[sentenceIdx][tokenIdx])

    print("test encodings completed")


if __name__ == "__main__":
    testEncodings()
