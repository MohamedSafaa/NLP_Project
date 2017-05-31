import nltk
from nltk import*
import codecs
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
f_in = open("Files/lexicon_dictionary.txt" , "r")

dic = {}
lis = []
for line in f_in :
	lis = line.split(" ")
	dic[lis[0]] = lis[1:len(lis)-1]



with codecs.open("Files/fearToken.txt", "r", "latin-1") as inputfile:
        text=inputfile.read()
inputfile.close()
sents = sent_tokenize(text)
words = []
stem = ""
temp = ""
overAllInput = []
snowball_stemmer = SnowballStemmer('english')
lancaster_stemmer = LancasterStemmer()
f_out = open("Files/featureVector.csv" , "w")
fearOut = [1 , 0 , 0]
tags = []
coun = 0
for sent in sents :
	fetVector = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0]
	words = word_tokenize(sent)
	tags = nltk.pos_tag(words)
	coun = 0
	for word in words :
		coun = coun + 1
		wordVector = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0]
		if (dic.has_key(word)):
			wordVector = dic[word]
		else :
			stem = lancaster_stemmer.stem(word)
			if(dic.has_key(stem)):
				wordVector = dic[stem]
			else :
				stem = snowball_stemmer.stem(word)
				if (dic.has_key(stem)):
					wordVector = dic[stem]
				else :
					temp = word + "er"
					if(dic.has_key(temp)and(tags[coun-1][1]=='VBP' or tags[coun-1][1]=='VB')):
						wordVector = dic[temp]
		for element in range(0,10) :
			fetVector[element] = fetVector[element] + int(wordVector[element])
	Str = ""
	for fet in fetVector :
		Str = Str + str(fet) + ","
			
	Str = Str+str(fearOut[0])+","+str(fearOut[1])+","+str(fearOut[2])+"\n" 
	f_out.write(Str)


# guilt

with codecs.open("Files/guiltToken.txt", "r", "latin-1") as inputfile:
        text=inputfile.read()
inputfile.close()
sents = sent_tokenize(text)
words = []
stem = ""
overAllInput = []
snowball_stemmer = SnowballStemmer('english')
fearOut = [0 , 1 , 0]
for sent in sents :
	fetVector = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0]
	words = word_tokenize(sent)
	tags = nltk.pos_tag(words)
	coun = 0
	for word in words :
		coun = coun + 1
		wordVector = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0]		
		if (dic.has_key(word)):
			wordVector = dic[word]
		else :
			stem = lancaster_stemmer.stem(word)
			if(dic.has_key(stem)):
				wordVector = dic[stem]
			else :
				stem = snowball_stemmer.stem(word)
				if (dic.has_key(stem)):
					wordVector = dic[stem]
				else :
					temp = word + "er"
					if(dic.has_key(temp)and(tags[coun-1][1]=='VBP' or tags[coun-1][1]=='VB')):
						wordVector = dic[temp]
		for element in range(0,10) :
			fetVector[element] = fetVector[element] + int(wordVector[element])
	Str = ""
	for fet in fetVector :
		Str = Str + str(fet) + ","
			
	Str = Str+str(fearOut[0])+","+str(fearOut[1])+","+str(fearOut[2])+"\n"
	f_out.write(Str)





# joy

with codecs.open("Files/joyToken.txt", "r", "latin-1") as inputfile:
        text=inputfile.read()
inputfile.close()
sents = sent_tokenize(text)
words = []
stem = ""
overAllInput = []
snowball_stemmer = SnowballStemmer('english')
fearOut = [0 , 0 , 1]
for sent in sents :
	fetVector = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0]
	words = word_tokenize(sent)
	tags = nltk.pos_tag(words)
	coun = 0
	for word in words :
		coun = coun + 1
		wordVector = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0]		
		if (dic.has_key(word)):
			wordVector = dic[word]
		else :
			stem = lancaster_stemmer.stem(word)
			if(dic.has_key(stem)):
				wordVector = dic[stem]
			else :
				stem = snowball_stemmer.stem(word)
				if (dic.has_key(stem)):
					wordVector = dic[stem]
				else :
					temp = word + "er"
					if(dic.has_key(temp)and(tags[coun-1][1]=='VBP' or tags[coun-1][1]=='VB')):
						wordVector = dic[temp]
		for element in range(0,10) :
			fetVector[element] = fetVector[element] + int(wordVector[element])
	Str = ""
	for fet in fetVector :
		Str = Str + str(fet) + ","
			
	Str = Str+str(fearOut[0])+","+str(fearOut[1])+","+str(fearOut[2])+"\n" 
	f_out.write(Str)

f_out.close()

