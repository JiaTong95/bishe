from settings import DATASET_PATH
import os
import time
import numpy as np
import pandas as pd
import gensim
import pickle
import random
import torch
import spacy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
from typing import List
from gensim.models.coherencemodel import CoherenceModel
import sys
sys.path.append('..')

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


class SpacyTokenizer(object):
    def __init__(self, lang="en", stopwords=None):
        self.stopwords = stopwords
        self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        print("Using SpaCy tokenizer")

    def tokenize(self, lines: List[str]) -> List[List[str]]:
        docs = self.nlp.pipe(lines, batch_size=1000)
        docs = [[token.lemma_ for token in doc if not (
            token.is_stop or token.is_punct)] for doc in docs]
        return docs


class DocDataset(Dataset):
    def __init__(self,
                 txtLines,
                 lang="en",
                 tokenizer=None,
                 stopwords=None,
                 no_below=5, no_above=0.1,
                 hasLable=False,
                 use_tfidf=False):
        tmpDir = f"temp"
        self.txtLines = txtLines
        self.dictionary = None
        self.bows, self.docs = None, None
        self.use_tfidf = use_tfidf
        self.tfidf, self.tfidf_model = None, None
        if not os.path.exists(tmpDir):
            os.mkdir(tmpDir)

        if stopwords == None:
            stopwords = set([l.strip('\n').strip() for l in open(
                f"{DATASET_PATH}/temp/stopwords-{lang}.txt", 'r', encoding='utf-8')])
        # self.txtLines is the list of string, without any preprocessing.
        # self.texts is the list of list of tokens.
        print('Tokenizing ...')
        if tokenizer is None:
            tokenizer = SpacyTokenizer(stopwords=stopwords)
        self.docs = tokenizer.tokenize(self.txtLines)
        self.docs = [line for line in self.docs if line != []]
        # build dictionary
        self.dictionary = Dictionary(self.docs)
        # self.dictionary.filter_n_most_frequent(remove_n=20)
        # use Dictionary to remove un-relevant tokens
        self.dictionary.filter_extremes(
            no_below=no_below, no_above=no_above, keep_n=None)
        self.dictionary.compactify()
        # because id2token is empty by default, it is a bug.
        self.dictionary.id2token = {v: k for k,
                                    v in self.dictionary.token2id.items()}
        # convert to BOW representation
        self.bows, _docs = [], []
        for doc in self.docs:
            _bow = self.dictionary.doc2bow(doc)
            if _bow != []:
                _docs.append(list(doc))
                self.bows.append(_bow)
        self.docs = _docs
        if self.use_tfidf == True:
            self.tfidf_model = TfidfModel(self.bows)
            self.tfidf = [self.tfidf_model[bow] for bow in self.bows]
        # serialize the dictionary
        gensim.corpora.MmCorpus.serialize(f"{tmpDir}/corpus.mm", self.bows)
        self.dictionary.save_as_text(f"{tmpDir}/dict.txt")
        pickle.dump(self.docs, open(f"{tmpDir}/docs.pkl", 'wb'))
        if self.use_tfidf:
            gensim.corpora.MmCorpus.serialize(
                f"{tmpDir}/tfidf.mm", self.tfidf)
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
            item = list(zip(*self.bows[idx]))
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt, bow

    def __len__(self):
        return self.numDocs

    def collate_fn(self, batch_data):
        texts, bows = list(zip(*batch_data))
        return texts, torch.stack(bows, dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def show_dfs_topk(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.dfs.items(
        )], key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(dfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ndoc} = {(1.0*freq/ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self, topk=20):
        ntokens = sum([v for k, v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.cfs.items(
        )], key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(cfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ntokens} = {(1.0*freq/ntokens):>.13f}')

    def topk_dfs(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0*dfs_topk[-1][-1]/ndoc


def get_topic_words(model, topn=15, n_topic=10, vocab=None, fix_topic=None, showWght=False):
    topics = []

    def show_one_tp(tp_idx):
        if showWght:
            return [(vocab.id2token[t[0]], t[1]) for t in model.get_topic_terms(tp_idx, topn=topn)]
        else:
            return [vocab.id2token[t[0]] for t in model.get_topic_terms(tp_idx, topn=topn)]
    if fix_topic is None:
        for i in range(n_topic):
            topics.append(show_one_tp(i))
    else:
        topics.append(show_one_tp(fix_topic))
    return topics


def calc_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words, []))
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div


def calc_topic_coherence(topic_words, docs, dictionary, emb_path=None, taskname=None, sents4emb=None, calc4each=False):
    # emb_path: path of the pretrained word2vec weights, in text format.
    # sents4emb: list/generator of tokenized sentences.
    # Computing the C_V score
    cv_coherence_model = CoherenceModel(
        topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_v')
    cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()

    # Computing the C_W2V score
    try:
        w2v_model_path = f"{DATASET_PATH}/temp/{taskname}/w2v_weight_kv.txt"
        # Priority order: 1) user's embed file; 2) standard path embed file; 3) train from scratch then store.
        if emb_path != None and os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                emb_path, binary=False)
        elif os.path.exists(w2v_model_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                w2v_model_path, binary=False)
        elif sents4emb != None:
            print('Training a word2vec model 20 epochs to evaluate topic coherence, this may take a few minutes ...')
            w2v_model = gensim.models.Word2Vec(
                sents4emb, size=300, min_count=1, workers=6, iter=20)
            keyed_vectors = w2v_model.wv
            keyed_vectors.save_word2vec_format(w2v_model_path, binary=False)
        else:
            raise Exception(
                "C_w2v score isn't available for the missing of training corpus (sents4emb=None).")

        w2v_coherence_model = CoherenceModel(
            topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_w2v', keyed_vectors=keyed_vectors)

        w2v_per_topic = w2v_coherence_model.get_coherence_per_topic() if calc4each else None
        w2v_score = w2v_coherence_model.get_coherence()
    except Exception as e:
        print(e)
        # In case of OOV Error
        w2v_per_topic = [None for _ in range(len(topic_words))]
        w2v_score = None

    # Computing the C_UCI score
    c_uci_coherence_model = CoherenceModel(
        topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_uci')
    c_uci_per_topic = c_uci_coherence_model.get_coherence_per_topic() if calc4each else None
    c_uci_score = c_uci_coherence_model.get_coherence()

    # Computing the C_NPMI score
    c_npmi_coherence_model = CoherenceModel(
        topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_npmi')
    c_npmi_per_topic = c_npmi_coherence_model.get_coherence_per_topic() if calc4each else None
    c_npmi_score = c_npmi_coherence_model.get_coherence()
    return (cv_score, w2v_score, c_uci_score, c_npmi_score), (cv_per_topic, w2v_per_topic, c_uci_per_topic, c_npmi_per_topic)


def mimno_topic_coherence(topic_words, docs):
    tword_set = set([w for wlst in topic_words for w in wlst])
    word2docs = {w: set([]) for w in tword_set}
    for docid, doc in enumerate(docs):
        doc = set(doc)
        for word in tword_set:
            if word in doc:
                word2docs[word].add(docid)

    def co_occur(w1, w2):
        return len(word2docs[w1].intersection(word2docs[w2]))+1
    scores = []
    for wlst in topic_words:
        s = 0
        for i in range(1, len(wlst)):
            for j in range(0, i):
                s += np.log((co_occur(wlst[i], wlst[j]) +
                            1.0)/len(word2docs[wlst[j]]))
        scores.append(s)
    return np.mean(s)


def evaluate_topic_quality(topic_words, test_data, taskname=None, calc4each=False):

    td_score = calc_topic_diversity(topic_words)
    print(f'topic diversity:{td_score}')

    (c_v, c_w2v, c_uci, c_npmi),\
        (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic) = \
        calc_topic_coherence(topic_words=topic_words, docs=test_data.docs, dictionary=test_data.dictionary,
                             emb_path=None, taskname=taskname, sents4emb=test_data, calc4each=calc4each)
    print('c_v:{}, c_w2v:{}, c_uci:{}, c_npmi:{}'.format(
        c_v, c_w2v, c_uci, c_npmi))
    scrs = {'c_v': cv_per_topic, 'c_w2v': c_w2v_per_topic,
            'c_uci': c_uci_per_topic, 'c_npmi': c_npmi_per_topic}
    if calc4each:
        for scr_name, scr_per_topic in scrs.items():
            print(f'{scr_name}:')
            for t_idx, (score, twords) in enumerate(zip(scr_per_topic, topic_words)):
                print(f'topic.{t_idx+1:>03d}: {score} {twords}')

    mimno_tc = mimno_topic_coherence(topic_words, test_data.docs)
    print('mimno topic coherence:{}'.format(mimno_tc))
    if calc4each:
        return (c_v, c_w2v, c_uci, c_npmi, mimno_tc, td_score), (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic)
    else:
        return c_v, c_w2v, c_uci, c_npmi, mimno_tc, td_score


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for pt in points:
        if smoothed_points:
            prev = smoothed_points[-1]
            smoothed_points.append(prev*factor+pt*(1-factor))
        else:
            smoothed_points.append(pt)
    return smoothed_points
