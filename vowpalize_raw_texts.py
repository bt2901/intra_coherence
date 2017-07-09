# encoding=utf8
import re
import os, glob, tarfile
from collections import Counter
import codecs, string
from nltk.corpus import stopwords


exclude = string.punctuation + u'«' + u'»' + u'—' + u'…'
digits = u'0123456789'
english_vowels = u'aeiouy'
regex = re.compile(u'[%s]' % re.escape(exclude)) # regex.sub('', s) to be used further

stop_words = []

forbidden_set = set(digits) | set(exclude) | set(stop_words)

def is_bad_word(word):
    if len(word) == 0:
        return True
    if word == u'topic' or word in forbidden_set:
        return True
    # are there punctuation marks inside word?
    if (len(forbidden_set.intersection(set(word))) > 0):
        return True
    return False

def strip_punctuation(word):
    return word.strip().strip(exclude)


    
def gen_vowpal(plaintext):
    all_tokens = []
    for line in plaintext.split(u"\n"):
        cur_tokens = (strip_punctuation(x) for x in line.strip().split(u" "))
        all_tokens += cur_tokens

    tokens_string = u" ".join(u"{}".format(tok) for tok in all_tokens if not is_bad_word(tok))
    return tokens_string




def vowpalize(dir_name):

    mask = os.path.join(dir_name, "*.txt")
    for doc in glob.glob(mask):
        with codecs.open(doc, "r", encoding="utf8") as f:
            content = f.read()
            doc_id = doc.strip().split("\\")[1]
            vowpal_desc = gen_vowpal(content)
            full_desc = u"{} |plain_text {}\n".format(doc, vowpal_desc)
            with codecs.open("vw_{}.txt".format("mixed"), "a", encoding="utf8") as out:
                out.write(full_desc)

vowpalize("PNaukaMixedLemmatized_short")
