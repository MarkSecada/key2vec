from key2vec.key2vec import Key2Vec
from key2vec.glove import Glove
from key2vec.docs import Document, Phrase

path = './data/glove.6B/glove.6B.50d.txt'
glove = Glove(path)
with open('./test.txt', 'r') as f:
    test = f.read()
m = Key2Vec(test, glove)
m.extract_candidates()
ranked = m.rank_candidates()
for row in ranked:
    print('{}. {}'.format(row.rank, row.text))