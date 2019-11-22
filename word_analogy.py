import os
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys

model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'
if len(sys.argv) > 1:
	if sys.argv[1] == 'nce':
		loss_model = 'nce'


model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
# print(dict(list(dictionary.items())[0:10]))
# print(steps)
# print(embeddings.view())
# print(embeddings.shape)
# print(len(list(dictionary.items())))
# exit(0)

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

mode = "average"
# mode = "max"
# mode = "min"


if loss_model == 'cross_entropy':
	output_file = "temp_dev_ce.txt"
elif loss_model == "nce":
	output_file = "temp_dev_nce.txt"



def getCosineSimilarity(v1, v2):
	cs = dot(v1, v2)/(norm(v1), norm(v2))
	return cs

with open("word_analogy_dev.txt") as fl:
	l = fl.readline()
	op_list = []
	while(l):
		exam, choi = l.strip().split("||")
		exam_pairs = exam.split(",")
		choi_pairs = choi.split(",")

		exam_cos = []
		choi_cos = []
		choi_words = []

		for pair in exam_pairs:
			context, target = pair.strip().split(":")
			context = context.replace('\"', '')
			target = target.replace('\"', '')
			context_id = dictionary[context]
			target_id = dictionary[target]
			context_embedding = embeddings[context_id]
			target_embedding = embeddings[target_id]
			cos_sim_list = getCosineSimilarity(v1 = context_embedding, v2 = target_embedding)
			cos_sim = sum(cos_sim_list)/len(cos_sim_list)
			exam_cos.append(cos_sim)

		if mode == "min":
			cos_similarity_exam = min(exam_cos)
		elif mode == "max":
			cos_similarity_exam = max(exam_cos)
		else:
			cos_similarity_exam = sum(exam_cos)/len(exam_cos)

		for pair in choi_pairs:
			c, t = pair.strip().split(":")
			c = c.replace('\"', '')
			t = t.replace('\"', '')
			c_id = dictionary[c]
			t_id = dictionary[t]
			c_embedding = embeddings[c_id]
			t_embedding = embeddings[t_id]
			cos_sim_list = getCosineSimilarity(v1 = c_embedding, v2 = t_embedding)
			cos_sim = sum(cos_sim_list)/len(cos_sim_list)
			choi_cos.append(cos_sim)
			choi_words.append(pair)

		diff = [abs(c - cos_similarity_exam) for c in choi_cos]
		min_index = diff.index(min(diff))
		max_index = diff.index(max(diff))
		word_str = ""
		for word in choi_words:
			word_str = word_str + word + " "
		word_str = word_str + choi_words[min_index] + " " + choi_words[max_index]
		op_list.append(word_str)

		l = fl.readline()

		# print(op_list)

		
fl.close()

op = open(output_file, "w")
for line in op_list:
	op.write(line)
	op.write("\n")

op.close()
