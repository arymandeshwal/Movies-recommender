import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating = 4.0)

#printtraining and testing data
print(repr(data['train']))
print(repr(data['test'])) 

# create model
model = LightFM(loss='warp')
#train model
model.fit(data['train'],epochs=30, num_threads = 2)

year = 1950

print("recommended:")

def sample_recommendation(model, data, user_ids, year):
	movies = list()
	#number of users and movies 
	n_users, n_items = data['train'].shape

	#generatte recommendations for each user we input

	for user_id in user_ids:
		#movies they already like
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movies our model predictrs they will like
		scores = model.predict(user_id, np.arange(n_items))
		#rank them in oder of most liked to least
		top_items = data['item_labels'][np.argsort(-scores)]

		#print out the results 
		#print("user "+str(user_id))
		#print("		known positives:")

		#for x in known_positives[:3]:
		#	print("			"+str(x))

		#print("		recommended:")
		for x in top_items[:2]:
			date = x[-5:-1]
			if (x not in movies) and (int(date) >= year):
				movies.append(str(x))
	print('\n'.join(movies))


lists = list(range(1,50))

sample_recommendation(model,data,lists,year)
"""
print("recommended:")
print(movies_re)
"""