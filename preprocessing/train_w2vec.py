from collections import defaultdict
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing.prepareTokenizer import prepare_vectorizer

train = np.load("data/train.npy")
test = np.load("data/test.npy")

# precompute vectorizer to obtain most frequent tokens
tfidf_vectorizer, tokenize_func = prepare_vectorizer(1) # now we just take all tokens (length 1)
X = tfidf_vectorizer.fit_transform(train)
voc_entries = tfidf_vectorizer.vocabulary_.items()
voc_words = [word for word, idx in voc_entries]

def w2v_transform(w2v, word2weight, data):
    return np.array([
        np.mean([w2v[w] * word2weight[w] for w in words if w in w2v] or
                [np.zeros(dim)], axis=0)
        for words in data
    ])

texts = []
test_texts = []

for doc in train:
    # only process the tokens recognized by the vectorizer
    texts.append([a for a in tokenize_func(doc) if a in voc_words])

for doc in test:
    # only process the tokens recognized by the vectorizer
    test_texts.append([a for a in tokenize_func(doc) if a in voc_words])

dim = 48
model = Word2Vec(texts, size=dim)
model.save("data/word2vec.model")
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

print("finished w2v training")

# get TF-IDF weights
tfidf = TfidfVectorizer(analyzer=lambda x: x, )
tfidf.fit(texts)

max_idf = max(tfidf.idf_)
word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

X = w2v_transform(w2v, word2weight, texts)
X_test = w2v_transform(w2v, word2weight, test_texts)

np.save("data/train_prepared_w2v.npy", X)
np.save("data/test_prepared_w2v.npy", X_test)