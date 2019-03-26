import key2vec

path = './data/glove.6B/glove.6B.50d.txt'
glove = key2vec.glove.Glove(path)
with open('./test.txt', 'r') as f:
    test = f.read()
m = key2vec.key2vec.Key2Vec(test, glove)
m.extract_candidates()
m.set_theme_weights()
m.build_candidate_graph()
ranked = m.page_rank_candidates()

for row in ranked:
    print('{}. {}'.format(row.rank, row.text))