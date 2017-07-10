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
    return tokens_string, all_tokens

def get_orig_labels(data_filtered, data):
    '''
    get original_topic_labels all at once
    could be sped up, but it takes < 1 second for entire collection, not really significant 
    '''
    original_topic_labels = [0 for x in data_filtered]
    i, j = 0, 0
    current_topic = None
    while i < len(data_filtered):
        if data[j] == "topic":
            current_topic = int(data[j+1]) - 1 # will crash if not number
            j += 2
        if data_filtered[i] == data[j]:
            original_topic_labels[i] = current_topic
            i, j = i+1, j+1
        else:
            j += 1
    return u" ".join(u"{}".format(tok) for tok in original_topic_labels)



def vowpalize(dir_name):

    mask = os.path.join(dir_name, "*.txt")
    for doc in glob.glob(mask):
        with codecs.open(doc, "r", encoding="utf8") as f:
            content = f.read()
            doc_id = doc.strip().split("\\")[1]
            doc_id = doc_id.split(".")[0]
            vowpal_desc, raw_tokens = gen_vowpal(content)
            original_labels = get_orig_labels(vowpal_desc.split(" "), raw_tokens)
            with codecs.open("vw_{}.txt".format("bimodal"), "a", encoding="utf8") as out:
                full_desc = u"{} |plain_text {} |labels {} \n".format(doc_id, vowpal_desc, original_labels)
                out.write(full_desc)
            with codecs.open("vw_{}.txt".format("plaintext"), "a", encoding="utf8") as out:
                out.write( u"{} |plain_text {} \n".format(doc_id, vowpal_desc) )
            with codecs.open("vw_{}.txt".format("labels"), "a", encoding="utf8") as out:
                out.write( u"{} |labels {} \n".format(doc_id, original_labels) )

vowpalize("PNaukaMixedLemmatized_short")
