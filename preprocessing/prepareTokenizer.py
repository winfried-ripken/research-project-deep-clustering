import csv

import unidecode
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import re
dateR = re.compile('\d{1,2}/\d{1,2}/{0,1}\d{0,4}')
timeR = re.compile('\d{1,2}:\d{1,2}:{0,1}\d{0,2}')
orderR = re.compile('32r\d*')
nR = [re.compile('navp-\d*'),
      re.compile('cpi-\d*'),
      re.compile('rrp-\d*'),
      re.compile('opi-\d*'),
      re.compile('cdeg\d*'),
      re.compile('3fw\d*')]

def parse_special_replacement(stemmer, stop_words, item, pattern):
    if item == pattern:
        return [pattern.lower()]
    elif pattern in item:
        start = item.find(pattern)
        end = start + len(pattern)

        return parse_token(stemmer, stop_words, item[:start]) + [pattern.lower()] + parse_token(stemmer, stop_words, item[end:])

    return []

def replace_n_regex(item):
    for i in range(len(nR)):
        if nR[i].match(item) is not None:
            return "____NO"+str(i)+"____"

    return item

def replace_regex(item):
    if dateR.match(item) is not None:
        return "____DATE____"
    elif timeR.match(item) is not None:
        return "____TIME____"
    elif orderR.match(item) is not None:
        return "____C_NO____"
    elif item.startswith("44l-retab"):
        return "____UID____"
    elif item.startswith("f0"):
        return "____F0_NO____"
    else:
        return replace_n_regex(item)

def remove_non_alpha_numerics(item):
    if len(item) == 0:
        return item

    if not item[0].isalnum():
        item = remove_non_alpha_numerics(item[1:])

    if len(item) == 0:
        return item

    if not item[-1].isalnum():
        item = remove_non_alpha_numerics(item[:-1])

    return item

def check_number(item):
    result = "____NUMBER____"

    work_i = item
    if "euro" in work_i or "e" in work_i:
        work_i = work_i.replace("euro", "")
        work_i = work_i.replace("e", "")
        result = "____PRICE____"

    if "ht" in work_i:
        work_i = work_i.replace("ht", "")
        result = "____NUMBER_HT____"

    if len(work_i) == 0:
        return item

    for c in work_i:
        if c == '-' or c == ',' or c == "." or c.isdigit():
            continue
        return item

    return result

def char_checks(item):
    if len(item) == 0:
        return item

    return check_number(replace_regex(remove_non_alpha_numerics(item)))

def test():
    print(char_checks("32r01000000821517531"))
    print(char_checks("\"123,76\""))
    print(char_checks("117.91"))
    print(char_checks("test"))
    print(char_checks("\"43,80\""))

def parse_token(stemmer, stop_words, item):
    if "'" in item:
        return [y for x in item.split("'") for y in parse_token(stemmer, stop_words, x)]

    patterns = ["____NOM___", "___TEL___", "____ADRESSE____",
                "____EMAIL____", "____PRENOM___", "____ZIPCODE___"]

    for pattern in patterns:
        special_replacements = parse_special_replacement(stemmer, stop_words, item, pattern)

        if len(special_replacements) > 0:
            return special_replacements

    item = char_checks(item)

    # remove stop words before stemming
    if stop_words.__contains__(item):
        return []

    # exceptions (no stemming) here
    if item == "obs":
        return [item]

    # standard stemming & remove accents
    item = stemmer.stem(unidecode.unidecode(item))

    # additional stems here
    if item == "infos":
        item = "info"
    elif item == "reo":
        item = "transfer"
    elif item == "transfert":
        item = "transfer"
    elif "resil" in item:
        item = "resil"

    # do not add single chars
    if len(item) <= 1:
        return []

    if item.__contains__('"'):
        print(item)
        print(char_checks(item))

    return [item]

def tokenize(stemmer, stop_words, text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        for token in parse_token(stemmer, stop_words, item):
            stems.append(token)
    return stems

# we would need this only for n-grams?
def prepare_vectorizer(max_n_gram_size, feature_threshold = 50000):
    vect = CountVectorizer(ngram_range=(1,max_n_gram_size))
    stemmer = FrenchStemmer()
    stop_words = set(stopwords.words('french'))

    stop_words.add("bonjour")
    stop_words.add("salut")
    stop_words.add("merci")
    stop_words.add("non")

    tfidf_vectorizer = TfidfVectorizer(analyzer=lambda text: vect._word_ngrams(tokenize(stemmer, stop_words, text)), max_features=feature_threshold)
    return tfidf_vectorizer, lambda text: tokenize(stemmer, stop_words, text)

def test_vectorizer():
    train = np.load("data/train.npy")
    tfidf_vectorizer, tokenize_func = prepare_vectorizer(1, 150000)
    X = tfidf_vectorizer.fit_transform(train)
    voc_entries = tfidf_vectorizer.vocabulary_.items()
    voc_words = [word for word, idx in voc_entries]

    with open('data/example_sentences.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(["text input", "processed output"])
        for line in train[:250]:
            writer.writerow([line, str([a for a in tokenize_func(line) if a in voc_words])])

    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in voc_entries]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    with open('data/most_frequent_terms.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(["token", "frequency score"])
        for line in words_freq:
            writer.writerow([line[0], line[1]])

    print("Length of vocabulary: " + str(len(tfidf_vectorizer.vocabulary_.items())))

if __name__ == "__main__":
    test_vectorizer()