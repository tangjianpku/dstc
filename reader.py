from __future__ import absolute_import

import os
import re
import numpy as np
from collections import Counter


EOS_TOKEN = "_eos_"

class TextReader(object):

    def __init__(self, config):

        self.data_path = "./data/%s" % config.dataset

        self.mem_size = config.memory_size
        self.sentence_size = config.sentence_size
        self.class_size = config.class_size

        train_path = os.path.join(self.data_path, "train_final.txt")
        test_path = os.path.join(self.data_path, "test_final.txt")
        total_path = os.path.join(self.data_path, "data_total.txt")

        vocab_path = os.path.join(self.data_path, "vocab.txt")

        self._build_vocab(total_path, vocab_path)

        self.train_data = self._file_to_data(train_path)
        self.test_data = self._file_to_data(test_path)

        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)

    def _read_text(self, file_path):
        with open(file_path) as f:
            return f.read().replace("\n", " %s " % EOS_TOKEN)

    def _build_vocab(self, file_path, vocab_path):

        text=self._read_text(file_path)
        tokens = [x.strip() for x in re.split('(\W+)?', text) if x.strip()]

        counter = Counter(tokens)

        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, counts = list(zip(*count_pairs))
        self.vocab = dict(zip(words, range(len(words))))

        self.vocab_counts = counts
        self.word_name = words
        self.vocab_size =len(self.vocab)

        #save_pkl(vocab_path, self.vocab)

    def _vectorize_sentence(self, sent):
        tokens = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

        tokens =tokens[0:self.sentence_size]

        lq = max(0, self.sentence_size - len(tokens))

        vec = [self.vocab[w.strip()] for w in tokens] + [0] * lq     

        return vec
    def _vectorize_class(self, class_id):

        #print class_id
        class_vec = np.zeros(self.class_size) 
        class_vec[int(class_id)] = 1

        return class_vec

    ################# overconfident ############## 
    def _file_to_data(self, file_path):
        
        lines = self._read_text(file_path).split(EOS_TOKEN)

        S = []
        Q = []
        A = []

        line_id = 0

        for line in lines:
          #data.append(np.array(map(self.vocab.get, text.split())))
          #print line
          if line ==" ":
            continue

          cols =line.split('\t')
          class_vec = [ ]
          query_vec  = [ ]
          contexts_vec = [ ]

          for idx, sentence in enumerate(cols):
            if idx == 0:
                class_id = sentence
                class_vec = self._vectorize_class(class_id)
                continue
            if idx == 1:
                query_vec = self._vectorize_sentence(sentence)
                continue

            context_vec = self._vectorize_sentence(sentence)

            contexts_vec.append(context_vec)
          #print len(cols)
          #print len(contexts_vec) 
          S.append(contexts_vec)
          Q.append(query_vec)
          A.append(class_vec)

        print "HHHHHHH", len(S[0])

        #print "S size:", len(S)

        return np.array(S), np.array(Q), np.array(A)
