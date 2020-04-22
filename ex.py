from nltk.corpus import stopwords

stop = stopwords.words("english")

if "don\'t" not in stop:
	print("yes")
else:
	print("no")