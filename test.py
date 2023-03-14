import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle

#df = pd.read_csv('SpamEmails/emails.csv')
#df.drop_duplicates(inplace = True)

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

#df['text'].head().apply(process_text)
#cv = CountVectorizer(analyzer=process_text,stop_words = "english", lowercase = True)
#messages_bow = cv.fit_transform(df['text'])
#X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)
#classifier = MultinomialNB()
#classifier.fit(X_train, y_train)
#cpickle.dump(classifier, open('naiveBayes.pkl', 'wb'))
#print('Predicted value: ',classifier.predict(X_test))
#print('done')
#cpickle.dump(cv.vocabulary_,open("feature.pkl","wb"))
#print('done')
classifier = cpickle.load(open('naiveBayes.pkl', 'rb'))
msg = "Subject: unbelievable new homes made easy  im wanting to show you this  homeowner  you have been pre - approved for a $ 454 , 169 home loan at a 3 . 72 fixed rate .  this offer is being extended to you unconditionally and your credit is in no way a factor .  to take advantage of this limited time opportunity  all we ask is that you visit our website and complete  the 1 minute post approval form  look foward to hearing from you ,  dorcas pittman"
#msg = "Subject: tuesday morning meeting first thing ? ? ?  vince :  i am sorry i couldnt connect with you last week . how would your tuesday  morning first thing , say 800 or 830 am be to get together to discuss the  demo proposal and other issues ? i can come by your office very conveniently  then . give me an email shout if you could squeeze it in on monday . i look  forward to speaking with you .  dale"

#msg = process_text(msg)
#print(msg)
cvv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("feature.pkl", "rb")))
cv1 = CountVectorizer(vocabulary=cvv.get_feature_names(),stop_words = "english", lowercase = True)
test = cv1.fit_transform([msg])
print('Predicted value: ',classifier.predict(test))


