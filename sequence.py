from __future__ import absolute_import
# -*- coding: utf-8 -*-
import numpy as np
from numpy import zeros, uint32
import random
from six.moves import range

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
        Pad each sequence to the same length:
        the length of the longest sequence.

        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.

        Supports post-padding and pre-padding (default).

        Parameters:
        -----------
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

        Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)

    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def make_sampling_table(tokenizer, sampling_factor=1e-5):
    '''
        This generates an array where the ith element
        is the probability that a word of rank i would be sampled,
        according to the sampling distribution used in word2vec.

        The word2vec formula is:
            p(word) = min(1, sqrt(word.frequency/sampling_factor) / (word.frequency/sampling_factor))

        We assume that the word frequencies follow Zipf's law (s=1) to derive
        a numerical approximation of frequency(rank):
           frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))
        where gamma is the Euler-Mascheroni constant.
    '''

    total_counts = sum(count for word,count in tokenizer.word_counts.iteritems())
    vocab_size = len(tokenizer.word_index)
    sampling_table = [0. for i in range(vocab_size+1) ]
    for word, widx in tokenizer.word_index.iteritems():
        frequnce = 1.0*tokenizer.word_counts[word]/total_counts
        sampling_table[widx] = min(1.0,(np.sqrt(frequnce/sampling_factor)+1.0)*(sampling_factor/frequnce))
    return sampling_table
    # gamma = 0.577
    # rank = np.array(list(range(size)))
    # rank[0] = 1
    # inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1./(12.*rank)
    # f = sampling_factor * inv_fq
    # return np.minimum(1., f / np.sqrt(f))

def make_neg_table(tokenizer, table_size=100000000, power=0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines.

        Called internally from `build_vocab()`.

        """

        print("constructing a table with noise distribution from %i words" % len(tokenizer.word_counts))
        # table (= list of words) of noise distribution for negative sampling
        vocab_size = len(tokenizer.word_counts)
        ng_sampling_table = zeros(table_size, dtype=uint32)

        if not vocab_size:
            print("empty vocabulary in word2vec, is this intended?")
            return

        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([count**power for word, count in tokenizer.word_counts.iteritems()]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        word2index = tokenizer.word_index
        #word2index = sorted(word2index.iteritems(), key=lambda d:d[1])
        tidx = 0
        d1 = 0.0
        for w, widx in word2index.iteritems():
            d1 += tokenizer.word_counts[w]**power / train_words_pow
            while tidx < table_size:
                ng_sampling_table[tidx] = widx
                tidx += 1
                if 1.0 * tidx / table_size > d1:
                    break
                if widx >= vocab_size:
                    widx = vocab_size - 1

        return ng_sampling_table

def skipgrams(sequence, vocabulary_size,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None, ng_sampling_table=None):
    '''
        Take a sequence (list of indexes of words),
        returns couples of [word_index, other_word index] and labels (1s or 0s),
        where label = 1 if 'other_word' belongs to the context of 'word',
        and label=0 if 'other_word' is ramdomly sampled

        Paramaters:
        -----------
        vocabulary_size: int. maximum possible word index + 1
        window_size: int. actually half-window. The window of a word wi will be [i-window_size, i+window_size+1]
        negative_samples: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
        categorical: bool. if False, labels will be integers (eg. [0, 1, 1 .. ]),
            if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]

        Returns:
        --------
        couples, lables: where `couples` are int pairs and
            `labels` are either 0 or 1.

        Notes:
        ------
        By convention, index 0 in the vocabulary is a non-word and will be skipped.
    '''
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_reduce = random.randint(0, window_size)
        window_start = max(0, i-window_size+window_reduce)
        window_end = min(len(sequence), i+window_size+1-window_reduce)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        if ng_sampling_table is None:
            ng_sampling_table = [1 for i in range(vocabulary_size)]
        nb_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)
        ng_tabel_size = len(ng_sampling_table)

        couples += [[words[i%len(words)], ng_sampling_table[random.randint(1, ng_tabel_size)]] for i in range(nb_negative_samples)]
        if categorical:
            labels += [[1,0]]*nb_negative_samples
        else:
            labels += [0]*nb_negative_samples

    if shuffle:
        seed = random.randint(0,10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels
