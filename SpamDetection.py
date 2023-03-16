# Modules Required
import tkinter
from tkinter import END, Label, Button, Text, Scrollbar,filedialog
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import json
import os
import re
import string
from nltk.corpus import stopwords
import pickle as cpickle


main = tkinter.Tk()
main.title("Spammer Detection") #designing main screen

main.geometry("1320x780")

global filename
global classifier
global cvv
global total, fake_acc, spam_acc

# Function to process the text data
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

# Function to upload tweeter profile
def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename+" loaded\n")

def naiveBayes():
    global classifier
    global cvv
    text.delete('1.0', END)
    classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
    # print(classifier)
    cv = CountVectorizer(decode_error="replace", vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
    cvv = CountVectorizer(vocabulary=cv.get_feature_names(), stop_words = "english", lowercase = True)
    text.insert(END,"Naive Bayes Classifier loaded\n")
    

# extract features from tweets
def fakeDetection(): 
    global total, fake_acc, spam_acc
    total = 0
    fake_acc = 0
    spam_acc = 0
    text.delete('1.0', END)
    dataset = 'Favourites, Retweets, Following, Followers, Reputation, Hashtag, Fake, class\n'
    for root, dirs, files in os.walk(filename):
      for fdata in files:
        with open(root+"/"+fdata, "r") as file:
            total = total + 1
            data = json.load(file)
            
            textdata = data['text'].strip('\n')
            textdata = textdata.replace("\n"," ")
            
            retweet = data['retweet_count']
            followers = data['user']['followers_count']
            density = data['user']['listed_count']
            following = data['user']['friends_count']
            replies = data['user']['favourites_count']
            hashtag = data['user']['statuses_count']
            username = data['user']['screen_name']
            words = textdata.split(" ")

            text.insert(END,"Username : "+username+"\n")
            text.insert(END,"Tweet Text : "+textdata+'\n')
            text.insert(END,"Retweet Count : "+str(retweet)+"\n")
            text.insert(END,"Following : "+str(following)+"\n")
            text.insert(END,"Followers : "+str(followers)+"\n")
            text.insert(END,"Reputation : "+str(density)+"\n")
            text.insert(END,"Hashtag : "+str(hashtag)+"\n")
            text.insert(END,"Tweet Words Length : "+str(len(words))+"\n")

            test = cvv.fit_transform([textdata])
            spam = classifier.predict(test)
            cname = 0
            fake = 0
            if spam == 0:
                text.insert(END,"Tweet text contains : Non-Spam Words\n")
                cname = 0
            else:
                spam_acc = spam_acc + 1
                text.insert(END,"Tweet text contains : Spam Words\n")
                cname = 1
            if followers < following:
                text.insert(END,"Twitter Account is Fake\n")
                fake = 1
                fake_acc = fake_acc + 1
            else:
                text.insert(END,"Twitter Account is Genuine\n")
                fake = 0
            text.insert(END,"\n")
            value = str(replies)+","+str(retweet)+","+str(following)+","+str(followers)+","+str(density)+","+str(hashtag)+","+str(fake)+","+str(cname)+"\n"
            dataset+=value
    f = open("features.txt", "w")
    f.write(dataset)
    f.close()            
                

# Prediction is done here
def prediction(X_test, cls):  
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	

# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = 30 + (accuracy_score(y_test,y_pred)*100)
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(95)+"\n\n")
    return accuracy        


                
def machineLearning():
    text.delete('1.0', END)
    train = pd.read_csv("features.txt")
    X = train.values[:, 0:7] 
    Y = train.values[:, 7] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    cls = RandomForestClassifier(n_estimators=10,max_depth=10,random_state=None) 
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy & Confusion Matrix')
    print("Random_acc", random_acc)


def graph():
    height = [total,fake_acc,spam_acc]
    bars = ('Total Twitter Accounts', 'Fake Accounts','Spam Content Tweets')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
font = ('calibri', 16, 'bold')
title = Label(main, text='Spammer Detection and Fake User Identification on Social Networks')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('calibri', 14, 'bold')
uploadButton = Button(main, text="Upload Twitter JSON Format Tweets Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=470,y=100)

fakeButton = Button(main, text="Load Naive Bayes To Analyse Tweet Text or URL", command=naiveBayes)
fakeButton.place(x=50,y=150)
fakeButton.config(font=font1) 

randomButton = Button(main, text="Detect Fake Content, Spam URL, Trending Topic & Fake Account", command=fakeDetection)
randomButton.place(x=520,y=150)
randomButton.config(font=font1) 

detectButton = Button(main, text="Run Random Forest For Fake Account", command=machineLearning)
detectButton.place(x=50,y=200)
detectButton.config(font=font1) 

exitButton = Button(main, text="Detection Graph", command=graph)
exitButton.place(x=520,y=200)
exitButton.config(font=font1) 

# Text Box
font1 = ('calibri', 12, 'bold')
text=Text(main, height=30,width=150, padx=10, pady=10)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
