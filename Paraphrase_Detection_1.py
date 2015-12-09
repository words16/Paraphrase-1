# -*- coding: utf-8 -*-

from scipy.stats.stats import pearsonr
import numpy as np
import nltk
import scipy.spatial.distance
from nltk.corpus import wordnet as wn
import math
from util.log import mylog

__author__ = 'yilee'

# an implementation of "A Semantic Similarity Approach to Paraphrase Detection"

stopwords = []
stopwords_filename = "/Volumes/Transcend/deep_learning/deep_learning_data/english"
msrp_train_file = "/Volumes/Transcend/deep_learning/deep_learning_data/MSRP/msr_paraphrase_train.txt"


def get_stopwords_list():
    global stopwords
    if len(stopwords) == 0:
        stopwords = open(stopwords_filename).read().split("\n")
    return stopwords


def process_line_msrp(line):
    words = line.split(" ")
    process_line = ""
    for w in words:
        if w.isalpha():
            if w.lower() not in get_stopwords_list():
                process_line += w
                process_line += " "
    return process_line.strip()


def similarity(s1, s2):
    s1 = process_line_msrp(s1)
    s2 = process_line_msrp(s2)

    # print "s1:", s1
    # print "s2:", s2

    s1_word = s1.split(" ")
    s2_word = s2.split(" ")

    union_s = list(set(s2_word).union(set(s1_word)))

    # print "union_s:", union_s

    n = len(union_s)
    s1_vector = np.zeros(n)
    s2_vector = np.zeros(n)

    i = 0
    for word in union_s:
        if word in s1_word:
            s1_vector[i] = 1
        if word in s2_word:
            s2_vector[i] = 1

        i += 1

    # print "s1_vector:", s1_vector
    # print "s2_vector:", s2_vector
    # print np.dot(s1_vector, s2_vector) / (np.linalg.norm(s1_vector) * np.linalg.norm(s2_vector))

    matrix_wup = np.zeros((n, n))

    i = 0
    j = 0
    for w1 in union_s:
        for w2 in union_s:
            if matrix_wup[i][j] > 0:
                j += 1
                continue

            if i == j:
                matrix_wup[i][j] = 1
                j += 1
                continue

            w1 = w1.lower()
            w2 = w2.lower()
            if w1 == w2:
                matrix_wup[i][j] = 1
                j += 1
                continue

            if not w1.isalpha():
                matrix_wup[i][j] = 0
                j += 1
                continue

            if not w2.isalpha():
                matrix_wup[i][j] = 0
                j += 1
                continue

            w1_synsets = wn.synsets(w1)
            if len(w1_synsets) == 0:
                matrix_wup[i][j] = 0
                j += 1
                continue

            w1_synset = w1_synsets[0]

            w2_synsets = wn.synsets(w2)
            if len(w2_synsets) == 0:
                matrix_wup[i][j] = 0
                j += 1
                continue
            w2_synset = w2_synsets[0]

            if w1_synset.pos() != w2_synset.pos():
                matrix_wup[i][j] = 0
                j += 1
                continue

            wup_sim = w1_synset.wup_similarity(w2_synset)
            if wup_sim < 0.8:
                wup_sim = 0.0

            try:
                a = float(wup_sim)
                if math.isnan(a):
                    print w1, w2
            except:
                print w1, w2

            matrix_wup[i][j] = wup_sim
            matrix_wup[j][i] = wup_sim

            j += 1
        i += 1
        j = 0

    similarity_value = np.dot(np.dot(s1_vector, matrix_wup), s2_vector.T) / (
        np.linalg.norm(s1_vector) * np.linalg.norm(s2_vector))
    #
    # print "matrix_wup:\n", matrix_wup
    #
    # print s1, "#", s2, ":", similarity_value
    mylog("1#" + str(similarity_value))
    mylog("\n")


if __name__ == '__main__':
    # for test
    # s1 = '''
    # dog eat cat
    # '''
    # s2 = '''
    # dog eat cat
    # '''

    # s1 = CorpusUtil.pre_process_msrp(" ".join(nltk.word_tokenize(s1)))
    # s2 = CorpusUtil.pre_process_msrp(" ".join(nltk.word_tokenize(s2)))

    i = 0
    for line in open(msrp_train_file).readlines():
        i += 1
        line = line.strip()
        split_line = line.split("\t")
        flag = int(split_line[0])
        s1 = split_line[3].strip()
        s2 = split_line[4].strip()

        s1 = " ".join(nltk.word_tokenize(s1))
        s2 = " ".join(nltk.word_tokenize(s2))

        mylog(str(flag) + " ")
        similarity(s1, s2)
        print i, "行......"
