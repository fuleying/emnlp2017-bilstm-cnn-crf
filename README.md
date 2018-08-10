# Introduction
- [CoNLL-2003 Shared Task](https://cogcomp.org/page/resource_view/81): Language-Independent Named Entity Recognition
- [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](http://www.aclweb.org/anthology/W03-0419.pdf)
- The CoNLL-2003 (Sang et al. 2003) shared task deals with language-independent named entity recognition as well (English and German).

# Dataset

The CoNLL-2003 shared task data files contain `four columns` separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a `word`, the second a `part-of-speech (POS) tag`, the third a `syntactic chunk tag` and the fourth the `named entity tag`. 
> The chunk tags and the named entity tags have the format `I-TYPE` which means that the word is inside a phrase of type TYPE. Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag `B-TYPE` to show that it starts a new phrase. A word with `tag O` is not part of a phrase.

The English data is a collection of news wire articles from the Reuters Corpus. The annotation has been done by people of the University of Antwerp. Because of copyright reasons we only make available the annotations. In order to build the complete data sets you will need access to the Reuters Corpus. It can be obtained for research purposes without any charge from NIST.

The German data is a collection of articles from the Frankfurter Rundschau. The named entities have been annotated by people of the University of Antwerp. Only the annotations are available here. In order to build these data sets you need access to the ECI Multilingual Text Corpus. It can be ordered from the Linguistic Data Consortium.

# Evaluation
- ![image](https://user-images.githubusercontent.com/6255387/43933988-674efbde-9c7f-11e8-890d-4c06c4681aa0.png)
- **Precision** is the percentage of named entities found by the learning system that are correct. 
- **Recall** is the percentage of named entities present in the corpus that are found by the system. 
- A named entity is correct only if it is an `exact match of the corresponding entity` in the data file.

## Results

| References              | Method                                              | F1    |
|-------------------------|-----------------------------------------------------|-------|
| [Ma 2016](https://arxiv.org/pdf/1603.01354.pdf)   | CNN-bidirectional LSTM-CRF | 91.21 |
| Luo et al. (2015)       | JERL                                                | 91.20 |
| Chiu et al. (2015)      | BLSTM-CNN + emb + lex                               | 91.62 |
| Huang et al. (2015)     | BI-LSTM-CRF                                         | 90.10 |
| Passos et al. (2014)    | Baseline + Gaz + LexEmb                             | 90.90 |
| Suzuki et al. (2011)    | L1CRF                                               | 91.02 |
| Collobert et al. (2011) | NN+SLL+LM2+Gazetteer                                | 89.59 |
| Collobert et al. (2011) | NN+SLL+LM2                                          | 88.67 |
| Ratinov et al. (2009)  | Word-class Model                                    | 90.80 |
| Lin et al. (2009)     | W500 + P125 + P64                                   | 90.90 |
| Ando et al. (2005)      | Semi-supervised approach                            | 89.31 |
| Florian et al. (2003)   | Combination of various machine-learning classifiers | 88.76 |

## References 

* **End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF** (ACL'16), Ma et al. [[pdf](https://arxiv.org/pdf/1603.01354.pdf)]
* **Named Entity Recognition with Bidirectional LSTM-CNNs** (CL'15), JPC Chiu et al. [[pdf](https://arxiv.org/pdf/1511.08308.pdf)]
* **Bidirectional LSTM-CRF Models for Sequence Tagging** (EMNLP'15), Z Huang et al. [[pdf](https://arxiv.org/pdf/1508.01991.pdf)]
* **Joint entity recognition and disambiguation** (EMNLP '15), G Luo et al. [[pdf](http://aclweb.org/anthology/D15-1104)]
* **Lexicon infused phrase embeddings for named entity resolution** (ACL'14), A Passos et al. [[pdf](http://www.aclweb.org/anthology/W14-1609)]
* **Learning condensed feature representations from large unsupervised data sets for supervised learning** (ACL'11), J Suzuki et al. [[pdf](http://www.aclweb.org/anthology/P11-2112)]
* **Natural Language Processing (Almost) from Scratch** (CL'11), R Collobert et al. [[pdf](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)]
* **Design Challenges and Misconceptions in Named Entity Recognition** (CoNLL'09), L Ratinov et al. [[pdf](http://www.aclweb.org/anthology/W09-1119)]
* **Phrase Clustering for Discriminative Learning** (ACL '09), D Lin et al. [[pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35520.pdf)]
* **A Framework for Learning Predictive Structures from Multiple Tasks and Unlabeled Data** (JMLR'05), RK Ando et al. [[pdf](http://www.jmlr.org/papers/volume6/ando05a/ando05a.pdf)]
* **Named Entity Recognition through Classifier Combination** (HLT-NAACL'03), R Florian et al. [[pdf](http://clair.si.umich.edu/clair/HLT-NAACL03/conll/pdf/florian.pdf)]
* **Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition** (CoNLL'03), EFTK Sang et al. [[pdf](http://aclweb.org/anthology/W03-0419)]

