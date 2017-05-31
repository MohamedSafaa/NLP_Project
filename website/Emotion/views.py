from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import numpy
from keras.models import model_from_json
import h5py
import nltk
from nltk import*
import codecs
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from django.http import HttpResponse
from django.shortcuts import render

def index(request):
	return render(request,'Emotion/index.html')

def GetEmotion(request):
	text = request.POST['Text']
	f_in = open("/home/saber/Desktop/NLP_Project/Files/lexicon_dictionary.txt", "r")

	dic = {}
	for line in f_in:
		lis = line.split(" ")
		dic[lis[0]] = lis[1:len(lis) - 1]
	sents = sent_tokenize(text)
	snowball_stemmer = SnowballStemmer('english')
	lancaster_stemmer = LancasterStemmer()
	f_out = open("/home/saber/Desktop/NLP_Project/TextfeatureVector.csv", "w")
	for sent in sents:
		fetVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		words = word_tokenize(sent)
		tags = nltk.pos_tag(words)
		coun = 0
		for word in words:
			coun = coun + 1
			wordVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			if (dic.has_key(word)):
				wordVector = dic[word]
			else:
				stem = lancaster_stemmer.stem(word)
				if (dic.has_key(stem)):
					wordVector = dic[stem]
				else:
					stem = snowball_stemmer.stem(word)
					if (dic.has_key(stem)):
						wordVector = dic[stem]
					else:
						temp = word + "er"
						if (dic.has_key(temp) and (tags[coun - 1][1] == 'VBP' or tags[coun - 1][1] == 'VB')):
							wordVector = dic[temp]
			for element in range(0, 10):
				fetVector[element] = fetVector[element] + int(wordVector[element])
		Str = ""
		for fet in fetVector:
			Str = Str + str(fet) + ","

		Str = Str[:-1]
		Str = Str + '\n'
		f_out.write(Str)
	f_out.close()

	dataset = numpy.loadtxt("/home/saber/Desktop/NLP_Project/TextfeatureVector.csv", delimiter=",")
	x = dataset[0:, 0:10]
	print(x)

	# Only code needed to  Load Code
	json_file = open("/home/saber/Desktop/NLP_Project/model.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("/home/saber/Desktop/NLP_Project/model.h5")
	print("Loaded model from disk")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')

	predictions = loaded_model.predict(x)
	print(predictions)
	rounded = numpy.around(predictions, decimals=0)
	print(rounded)

	FearCount=0
	GuiltCount=0
	JoyCount=0

	FearPharses = ''
	GuiltPharses = ''
	JoyPharses = ''

	pharse = 1

	for round in rounded:
		if str(round) == '[ 1.  0.  0.]':
			FearCount = FearCount + 1
			FearPharses = FearPharses + str(pharse)+','
			print('fear')
		elif str(round) == '[ 0.  1.  0.]':
			GuiltCount = GuiltCount + 1
			GuiltPharses = GuiltPharses + str(pharse)+','
			print('guilt')
		elif str(round) == '[ 0.  0.  1.]':
			JoyCount = JoyCount + 1
			JoyPharses = JoyPharses + str(pharse)+','
			print('joy')
		pharse = pharse + 1


	context = {
		'FearCount':FearCount,
		'FearPharses':FearPharses,
		'GuiltCount': GuiltCount,
		'GuiltPharses': GuiltPharses,
		'JoyCount':JoyCount,
		'JoyPharses':JoyPharses,
	}

	return render(request, 'Emotion/Emotions.html',context)
