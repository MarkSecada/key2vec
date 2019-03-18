from key2vec.key2vec import Key2Vec
from key2vec.glove import Glove
from key2vec.docs import Document, Phrase

path = './data/glove.6B/glove.6B.50d.txt'
glove = Glove(path)
m = Key2Vec("Hello. My name is Mark Secada. I'm a Data Scientist!",
    glove)
m.extract_candidates()
m.rank_candidates()