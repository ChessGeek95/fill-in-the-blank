__author__ = "A.V"

from hazm import *
from copy import deepcopy as cpy
import os.path
import gensim, logging

tagger = POSTagger(model='resources/postagger.model')
chunker = Chunker(model='resources/chunker.model')

class Sentence:
    def __init__(self, text):
        self.text = text
        self.tagged_sent_S = tagger.tag(word_tokenize(text))
        self.tagged_sent_T = tagger.tag(word_tokenize(text.replace('$','ددد')))
        self.blank_i = self.tagged_sent_S.index('$')
        self.blank_tag = []
        self.pre_blank = [s for s in self.tagged_sent_S[self.blank_i-2:self.blank_i]]
        self.post_blank = [s for s in self.tagged_sent_S[self.blank_i+1:self.blank_i+3]]

    def blank_tag_prob(self ,tag):
        if tag == "Noun":
            return noun_prob_of_blank(self)
        elif tag == "Verb":
            return verb_prob_of_blank(self)
        elif tag == "Adj":
            return adj_prob_of_blank(self)
        elif tag == "Adv":
            return adv_prob_of_blank(self)


def noun_prob_of_blank(sent):  # is blank Noun?
    p = 0
    if "N" in sent.tagged_sent_T[blank_i][1]:
	p+= 15
    if len(tree2brackets(sent.tagged_sent_T)) != len(tree2brackets(sent.tagged_sent_T)):
	p+= 12
    if len([s for s in sent.tagged_sent_T if "V" in s[1]])== 0:
        p-=10
    return p

def verb_prob_of_blank(sent):    # is blank verb?
    p = 0
    if len([s for s in sent.tagged_sent_T if "V" in s[1]])== 0 or \
	len([s for s in sent.tagged_sent_S if "V" in s[1]])== 0:
	p+=5
    if len([s for s in sent.tagged_sent_T if "V" in s[1]])== 0 and \
	len([s for s in sent.tagged_sent_S if "V" in s[1]])== 0:
    	p+=10
    return p

def adv_prob_of_blank(sent):   # is blank Adverb?
    p = 0
    if len(tree2brackets(sent.tagged_sent_T)) == len(tree2brackets(sent.tagged_sent_S)):
	p+= 3
    if sent.blank_i == 0 or sent.blank_i == len(sent.tagged_sent_S)-1:
        p+=2
    if len(tree2brackets(tagger.tag(word_tokenize(sent.tagged_sent_S.replace('$','')))))\
	== len(tree2brackets(sent.tagged_sent_T)):
	p+=4
    return p
        
def adj_prob_of_blank(sent):   # is blank Adjective?
    p = 0
    if len(tree2brackets(sent.tagged_sent_T)) == len(tree2brackets(sent.tagged_sent_S)):
	p+= 5
    if len(sent.pre_blank)>0:
        if ("N" in sent.pre_blank[0][1]):
            p += 3
    return p


def train_word2vec(corpus_name ,trained_model_name):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    f = open('corpus_name' ,'r')
    sentences = []
    for l in f:
        ll = l.split(' ')
        ll = [i.lower() for i in ll if len(i)>1]
        sentences.append(ll)
    model = gensim.models.Word2Vec(sentences, size=150, window=7, min_count=5)
    model.save(trained_model_name)
    return model

def load_word2vec_model(trained_model_name):
    return gensim.models.Word2Vec.load(trained_model_name)



types = ["Noun" ,"Verb" ,"Adj" ,"Adv"]
signs = dict()
signs["Noun"] = ["NN" ,"NNP" ,"NNPS" ,"NNS" ,"PRP" ,"PRP$"]
signs["Verb"] = ["VB" ,"VBP" ,"VBZ" ,"VBD" ,"VBG" ,"VBN"]
signs["Adj"] = ["ADJ" ,"JJ","JJR" ,"JJS"]
signs["Adv"] = ["ADV" ,"RB" ,"RBR" ,"RBS"]

sentence = str((input("Enter a sentence:")))
Sent = Sentence(sentence)

most_prob = "Noun"
prob = 0
for t in types:
    if Sent.blank_tag_prob(t) > prob:
        most_prob = t
        prob = Sent.blank_tag_prob(t)


answers = []
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = nltk.collocations.TrigramCollocationFinder.from_words(nltk.corpus.brown.words())
finder1 = cpy(finder)
if len(Sent.pre_blank)>1:
    finder1.apply_ngram_filter(lambda w1, w2, w3: \
                              (Sent.pre_blank[0][0],Sent.pre_blank[1][0])!=(w1, w2))
    L1 = [s[0] for s in finder1.score_ngrams(trigram_measures.raw_freq)[:10]]
    for i in L1:
        if nltk.pos_tag([i[2]])[0][1] in signs[most_prob]:
            answers.append(i[2])
    
finder2 = cpy(finder)
if len(Sent.pre_blank)>0 and len(Sent.post_blank)>0:
    finder2.apply_ngram_filter(lambda w1, w2, w3: \
                              (Sent.pre_blank[1][0],Sent.post_blank[0][0])!=(w1, w3))
    L2 = [s[0] for s in finder2.score_ngrams(trigram_measures.raw_freq)[:10]]
    for i in L2:
        if nltk.pos_tag([i[1]])[0][1] in signs[most_prob]:
            answers.append(i[1])

finder3 = cpy(finder)
if len(Sent.post_blank)>1:
    finder3.apply_ngram_filter(lambda w1, w2, w3: \
                              (Sent.post_blank[0][0],Sent.post_blank[1][0])!=(w2, w3))
    L3 = [s[0] for s in finder3.score_ngrams(trigram_measures.raw_freq)[:10]]
    for i in L3:
        if nltk.pos_tag([i[0]])[0][1] in signs[most_prob]:
            answers.append(i[0])

if len(answers) == 0:
    finder4 = cpy(finder)
    if len(Sent.pre_blank)>1:
        finder4.apply_ngram_filter(lambda w1, w2, w3: \
                                  Sent.pre_blank[0][0] not in (w1, w2))
        L4 = [s[0] for s in finder4.score_ngrams(trigram_measures.raw_freq)[:10]]
        for i in L4:
            if nltk.pos_tag([i[2]])[0][1] in signs[most_prob]:
                answers.append(i[2])
            if nltk.pos_tag([i[1]])[0][1] in signs[most_prob]:
                answers.append(i[1])


if os.path.exists("trained_model"):
    model = load_word2vec_model("trained_model")
else:
    model = train_word2vec("train.txt" ,"trained_model")

for i in answers:
    most_similars = [s[0] for s in model.most_similar(i) if nltk.pos_tag([s[0]])[0][1] in signs[most_prob]][:3]
    for j in most_similars:
        if nltk.pos_tag(j)[1] in signs[most_prob]:
            answers.append(j)

text = Sent.text
for i in answers:
    print(text.replace('$',i))
