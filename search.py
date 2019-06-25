import os
import re
import math
import json
import codecs
import collections
from itertools import islice
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
lancaster=LancasterStemmer()
def stopword(data_path):
	stopword_file =  codecs.open(data_path, "r")
	stopwords = stopword_file.read().split('\n')
	stopword_file.close()
	return stopwords

def build_inverted_index(data_path, stopwords):
	file = os.listdir(data_path)
	dictionary = {}
	non_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W")
	for filename in file:
		link = codecs.open(data_path + "/" +filename, "r")
		split_words = link.read().lower()
		link.close()
		split_words = re.sub(non_words,r' ', split_words)
		split_words = re.sub(r'[-|?|$|.|!|"|,|(|)|/|_|\'|`|*|+|@|#|%|^|&|[|]|{|}|;|:|<|>|،|、|…|⋯|᠁|ฯ|‹|›|«|»|‘|’|“|”|‱|‰|±|∓|¶|‴|§|‖|¦|©|🄯|℗|®|℠|™|]',r' ', split_words)
		new = list(split_words.split())
		new_stem = list()
		for word in new:
			new_stem.append(lemmatizer.lemmatize(word))
			# new_stem.append(porter.stem(word))
		new = new_stem
		for word in new:
			if(word not in dictionary):
				dictionary[word] = {}
			if(filename not in dictionary[word]):
				dictionary[word][filename] = 1
			else:
				dictionary[word][filename] += 1

	return dictionary

def build_indexing(inverted_index):
	indexing = {}
	for item in inverted_index:
		tf_overall = 0
		test = 0
		indexing[item] = {}
		indexing[item]['posting_list'] = {}
		# print(inverted_index)
		for doc in inverted_index[item]:
			tf_overall += inverted_index[item][doc]
			test += 1
			indexing[item]['posting_list'][doc] = {}
			indexing[item]['posting_list'][doc]['tf'] = inverted_index[item][doc]
		indexing[item]['tf_overall'] = tf_overall
		indexing[item]['num_of_docs'] = len(inverted_index[item])
		indexing[item]['idf'] = 1/indexing[item]['num_of_docs']
	# norm
	normalize = {}
	files = os.listdir('Cranfield')
	for file in files:
		temp = 0
		for item in indexing:
			if file in indexing[item]['posting_list'].keys():
				temp += pow(indexing[item]['idf']*indexing[item]['posting_list'][file]['tf'],2)
		normalize[file] = math.sqrt(temp)
		# print(normalize[file])
	for item in indexing:
		for doc in indexing[item]['posting_list']:
			indexing[item]['posting_list'][doc]['w'] = indexing[item]['posting_list'][doc]['tf']*indexing[item]['idf']/normalize[doc]
	
	return indexing
def queryprocess(query, non_words,indexing):
	
	non_words = re.compile(r"\b(" + "|".join(non_words) + ")\\W")
	query = re.sub(non_words,r' ', query.lower())
	query = (re.sub(r'[-|?|$|.|!|"|,|(|)|/|_|\'|`|*|+|@|#|%|^|&|[|]|{|}|;|:|<|>|،|、|…|⋯|᠁|ฯ|‹|›|«|»|‘|’|“|”|‱|‰|±|∓|¶|‴|§|‖|¦|©|🄯|℗|®|℠|™|]',r' ', query).split())
	# print(query)
	query_stem = list()
	for word in query:
		query_stem.append(lemmatizer.lemmatize(word))
		# query_stem.append(porter.stem(word))
	query= query_stem
	# print(query)
	tf_query = {}
	for word in query:
		if word not in tf_query.keys():
			tf_query[word] = 1
		else:
			tf_query[word] +=1
	# print(tf_query)

	temp = 0
	for word in tf_query:
		# temp += pow(indexing[word]['idf']*tf_query[word],2)
		if word in indexing.keys():
			temp += pow(indexing[word]['idf']*tf_query[word],2)
	normalize = math.sqrt(temp)
	w_query = {}
	for word in tf_query:
		if word in indexing.keys():
			w_query[word] = tf_query[word]*indexing[word]['idf']/normalize
	# print(normalize)
	# print(w_query)

	similarity = {}
	files = os.listdir('Cranfield')
	for file in files:
		temp_simalar = 0
		for word_query in w_query.keys():
			if word_query in indexing.keys():
				if file in indexing[word_query]['posting_list'].keys():
					temp_simalar += w_query[word_query]*indexing[word_query]['posting_list'][file]['w']
		similarity[file] = temp_simalar
	result = dict(collections.OrderedDict(sorted(similarity.items(), key = lambda kv: kv[1], reverse = True)))
	file_relevance = list(islice(result, 50))
	final_result = {}
	for key in file_relevance:
		final_result[key] = result[key]
	# print(final_result)
	return final_result


# def searchQuery(query, invert_begin):	
# 	stopwords = stopword("stopwords.txt")
# 	indexing = build_indexing(build_inverted_index("Cranfield/", stopwords))
# 	return queryprocess(query,stopwords,indexing)

# searchQuery('what are the structural and aeroelastic problems associated with flight of high speed aircraft')



def computeMAP():
	stopwords = stopword("stopwords.txt")
	indexing = build_indexing(build_inverted_index("Cranfield/", stopwords))

	listRs = os.listdir('DEV/RES')
	listfilesCheck = []
	for path in listRs:
		temp = int(path[:-4])
		listfilesCheck.append(temp)

	listfilesCheck.sort()
	# print(len(listfilesCheck))

	# print(rs)
	i = 0
	totalPrecisionOfAllDoc = 0
	averagePrecision = 0
	precision = {}
	recall = {}
	query_index = 1
	with open('DEV/query.txt') as fileQr:
		queries = fileQr.readlines()
		for q in queries:
			if (len(q.strip()) == 0 or q == '\n'):
				queries.remove(q)
				if(queries[-1] == '\n'):
					del queries[-1]
		# print('do dai query')
		# print(len(queries))
		# print(queries)
		for query in queries:
			result = queryprocess(query[2:-2],stopwords,indexing)
			self_result = [key[:-4] for key in result.keys()]
			
			true_result = []
			precision[query_index] = []
			recall[query_index] = []
			path = 'DEV/RES/' + str(listfilesCheck[i]) + '.txt'
			i += 1
			with open(path) as f:
				lines = f.readlines()
				for line in lines:
					key = line.split() #key moi ket qua cua thay
					true_result.append(str(key[1]))
				# print("ket qua cua thay" + str(len(true_result)))
				# print(true_result)
				total_precision = 0
				currentResultIndex = 0
				currentReleventIndex = 0
				self_result = self_result[:len(true_result)]
				
				# write file
				with open('result/' + str(i) + '.txt', 'w') as f:
					for item in self_result:
						f.write("%s\n" % item)



				# print("ket qua cua minh")
				# print(self_result)
				for doc in self_result:
					if doc in true_result:
						currentReleventIndex += 1
						currentResultIndex +=1
						total_precision += currentReleventIndex/currentResultIndex
						current_precision = currentReleventIndex/currentResultIndex
						current_recall = currentReleventIndex/len(true_result)
						precision[query_index].append(current_precision)
						recall[query_index].append(current_recall)
					else:
						currentResultIndex +=1
				if(currentReleventIndex):
					totalPrecisionOfAllDoc += total_precision / currentReleventIndex
			query_index += 1
	averagePrecision = totalPrecisionOfAllDoc/len(listfilesCheck)


	with open('precision.txt', 'w') as f:
		json.dump(precision, f)
	with open('recall.txt', 'w') as f:
		json.dump(recall, f)
	with open('MAP.txt', 'w') as f:
		f.write("MAP:\t%s" % averagePrecision)
	return precision, recall, averagePrecision


if __name__ == '__main__':

# bỏ cmt này để test từng câu truy vấn

	# non_words = stopword("stopwords.txt")
	# inverted_index = build_inverted_index('./Cranfield', non_words)
	# indexing = build_indexing(inverted_index)

	# thay chuỗi trong hàm queryprocess để test query khác
	# result = queryprocess('what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft', non_words, indexing)
	# print(result)
	computeMAP()