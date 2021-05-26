import nltk

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print('tokens: ', tokens)

tagged = nltk.pos_tag(tokens)
print('tagged6: ',tagged[0:6])

entities = nltk.chunk.ne_chunk(tagged)
print('entities : ',entities)

from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
print('Arbre des mots:', t.draw())