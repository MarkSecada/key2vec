from key2vec.key2vec import Key2Vec

path = './data/glove.6B/glove.6B.50d.txt'
with open('test.txt', 'r') as f:
    article = f.read()
m = Key2Vec(article, path, 50)
candidates = m.extract_candidates()
candidates = list(candidates.keys())
rankings = m.rank_candidates(candidates)

final_cand = []
for candidate in candidates:
    final_cand.append([candidate, rankings[candidate]['score']])