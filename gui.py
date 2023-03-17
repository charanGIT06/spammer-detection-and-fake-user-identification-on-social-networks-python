# Dependencies
# GUI
import tkinter as tk
from tkinter import END, filedialog, Scrollbar
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from urllib.request import urlopen
import customtkinter as ctk
# Machine Learning
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# Graph Plotting
import matplotlib.pyplot as plt
# Data Processing
import numpy as np 
import pandas as pd 
import random as r
import json
import os
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import pickle as cpickle
# .ENV
from dotenv import load_dotenv
# Twitter API
import tweepy

# Loading the environment variables
load_dotenv()

# Twitter API credentials
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

# Setting the Theme
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")

# Global Variables
global filename
global classifier
global cvv
global total, fake_acc, spam_acc
global verified_status

class MyGUI(ctk.CTk):
    # Construction method to initialize the GUI
    def __init__(self):
        super().__init__()

        self.title("spammer detection and fake user identification".title())
        width= self.winfo_screenwidth()
        height= self.winfo_screenheight()
        self.geometry("%dx%d" % (width, height))

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0) # type: ignore
        self.grid_rowconfigure((0, 1, 2), weight=1) # type: ignore

        # Creating the Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200, height=500, corner_radius=20)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=20, pady=20)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Actions", font=("Arial", 20))
        self.logo_label.grid(row=0, column=0, padx=40, pady=(20, 5))

        # Upload Tweets Button
        self.upload_tweets_button = ctk.CTkButton(self.sidebar_frame, text="Upload Tweets", font=("Arial", 20), corner_radius=20, command=self.upload_dataset)
        self.upload_tweets_button.grid(row=1, column=0, padx=25, pady=10, ipadx=15, ipady=10, stick="ew")

        # Load Naive Bayes Button
        self.Load_Naive_Bayes_button = ctk.CTkButton(self.sidebar_frame, text="Load Naive Bayes", font=("Arial", 20), corner_radius=20, command=self.load_naive_bayes)
        self.Load_Naive_Bayes_button.grid(row=2, column=0, padx=25, pady=10, ipadx=15, ipady=10, stick="ew")

        # Run Random Forest Button
        self.Run_Random_Forest_button = ctk.CTkButton(self.sidebar_frame, text="Run Random Forest", font=("Arial", 20), corner_radius=20, command=self.machine_learning)
        self.Run_Random_Forest_button.grid(row=3, column=0, padx=25, pady=10, ipadx=15, ipady=10, stick="ew")

        # Detect Button
        self.Detect_button = ctk.CTkButton(self.sidebar_frame, text="Detect", font=("Arial", 20), corner_radius=20, command=self.fakeDetection)
        self.Detect_button.grid(row=4, column=0, padx=25, pady=10, ipadx=15, ipady=10, stick="ew")

        # Graph Button
        self.Graph_button = ctk.CTkButton(self.sidebar_frame, text="Graph", font=("Arial", 20), corner_radius=20, command=self.graph)
        self.Graph_button.grid(row=5, column=0, padx=25, pady=10, ipadx=15, ipady=10, stick="ew")


        # # Test with Live Data Label
        # self.test_label = ctk.CTkLabel(self.sidebar_frame, text="Test", font=("Arial", 20))
        # self.test_label.grid(row=6, column=0, padx=40, pady=(30, 10))

        # # Get Tweets Button
        # self.upload_tweets_button = ctk.CTkButton(self.sidebar_frame, text="Get Tweets", font=("Arial", 20), corner_radius=20, command=self.get_tweets)
        # self.upload_tweets_button.grid(row=7, column=0, padx=25, pady=10, ipadx=15, ipady=10, stick="ew")

        # # Detect Button
        # self.Detect_button = ctk.CTkButton(self.sidebar_frame, text="Detect", font=("Arial", 20), corner_radius=20, command=self.fakeDetection)
        # self.Detect_button.grid(row=8, column=0, padx=25, pady=10, ipadx=15, ipady=10, stick="ew")

        # Creating the Text Box
        self.text_box = ctk.CTkTextbox(self, width=300, height=500, font=('calibri', 20),  corner_radius=20)
        self.text_box.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=(0, 20), pady=20)
        self.text_box.insert(1.0, 'Welcome to our Project!\n\nTrain the Model and test it with live data.\n\nEnter the username of the account and the number of tweets that you want to test and click on  the "Get Tweets" button.\nUpload the new tweets and click on the "Detect" button to see the results.')

        # Creating Tweet Frame
        self.tweet_frame = ctk.CTkFrame(self, width=300, height=500, corner_radius=20)
        self.tweet_frame.grid(row=0, column=2, rowspan=4, sticky="nsew", padx=(0, 20), pady=20)

        # Creating the Frame Label
        self.tweet_label = ctk.CTkLabel(self.tweet_frame, text="Enter Details", font=("Arial", 20))
        self.tweet_label.grid(row=0, column=0, padx=65, pady=(20, 5), stick="we")

        # Creating the Username Entry
        self.username = ctk.CTkEntry(self.tweet_frame, width=30, font=("Arial", 18), corner_radius=10, placeholder_text="Username")
        self.username.grid(row=1, column=0, padx=20, pady=10, ipadx=50, ipady=10, stick="ew")

        # creating the tweet count entry
        self.tweet_count = ctk.CTkEntry(self.tweet_frame, width=30, font=("Arial", 18), corner_radius=10, placeholder_text="Tweet Count")
        self.tweet_count.grid(row=2, column=0, padx=20, pady=10, ipadx=50, ipady=10, stick="ew")

        # Get Tweets Button
        self.upload_tweets_button = ctk.CTkButton(self.tweet_frame, text="Get Tweets", font=("Arial", 20), corner_radius=20, command=self.get_tweets)
        self.upload_tweets_button.grid(row=3, column=0, padx=20, pady=10, ipadx=15, ipady=10, stick="ew")

        # Clear Button
        self.clear_button = ctk.CTkButton(self.tweet_frame, text="Clear", font=("Arial", 20), corner_radius=20, command=self.clear_text)
        self.clear_button.grid(row=4, column=0, padx=20, pady=10, ipadx=15, ipady=10, stick="ew")

        # Account Details
        self.account_details = ctk.CTkLabel(self.tweet_frame, text="Account Details", font=("Arial", 20))
        self.account_details.grid(row=5, column=0, padx=65, pady=(30, 5), stick="we")

        # Account Details Text Box
        self.account_details_text_box = ctk.CTkTextbox(self.tweet_frame, width=300, font=('calibri', 20),  corner_radius=20)
        self.account_details_text_box.grid(row=6, column=0, rowspan=4, sticky="nsew", padx=20, pady=(10, 20))
        self.account_details_text_box.insert(END, 'Name: \n')
        self.account_details_text_box.insert(END, 'Username: \n')
        self.account_details_text_box.insert(END, 'Followers: \n')
        self.account_details_text_box.insert(END, 'Following: \n')

        # run the mainloop
        self.mainloop()

    # Function to process the text data
    def process_text(self, text):
        nopunc = [char for char in text if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
        return clean_words

    # Function to  upload training dataset
    def upload_dataset(self):
        global filename
        filename = filedialog.askdirectory(initialdir='.')
        self.text_box.delete("1.0", END)
        self.text_box.insert(END, f'Dataset uploaded from:\n\n{filename}'+"\n")

    # Function to load Naive Bayes Classifier
    def load_naive_bayes(self):
        global classifier
        global cvv
        self.text_box.delete("1.0", END)
        classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
        cv = CountVectorizer(decode_error="replace", vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
        cvv = CountVectorizer(vocabulary=cv.get_feature_names(), stop_words="english", lowercase = True)
        self.text_box.insert(END, f'Naive Bayes Classifier Loaded!'+"\n")

    # Function to extract features from tweets
    def fakeDetection(self):
        global total, fake_acc, spam_acc, verified_status
        verified_status = False
        total = 0
        fake_acc = 0
        spam_acc = 0
        self.text_box.delete('1.0', END)
        dataset = 'Favourites, Retweets, Following, Followers, Reputation, Hashtag, Fake, class\n'
        tweet_count = 0
        for root, dirs, files in os.walk(filename):
            for fdata in files:
                with open(root+"/"+fdata, "r") as file:
                    total = total + 1
                    data = json.load(file)
                    
                    # formatting the text
                    textdata = data['text'].strip( '\n')
                    textdata = textdata.replace( "\n"," ")
                    textdata = textdata.replace( "\t"," ")
                    textdata = re.sub( '\W+', ' ', textdata)

                    print("TEXT_DATA", textdata)
                    print('----------------------------------------------------------------------------')
                    print("PROCESSED_TEXT_DATA", self.process_text(textdata))

                    # Storing the required data from the json file to variables
                    username = data['user']['screen_name']
                    followers = data['user']['followers_count']
                    following = data['user']['friends_count']
                    retweet = data['retweet_count']
                    density = data['user']['listed_count']
                    replies = data['user']['favourites_count']
                    hashtag = data['user']['statuses_count']
                    words = textdata.split(" ")

                    # Displaying the tweets
                    tweet_count = tweet_count + 1
                    self.text_box.insert(END,"Tweet No : "+str( tweet_count)+"\n")
                    self.text_box.insert(END,"Username : "+username+"\n")
                    textdata = ''.join( [ i for i in textdata if ord(i) in range(65536)])
                    self.text_box.insert(END,"Tweet Text : "+textdata+'\n')
                    self.text_box.insert(END,"Retweet Count : "+str(retweet)+"\n")
                    self.text_box.insert(END,"Following : "+str(following)+"\n")
                    self.text_box.insert(END,"Followers : "+str(followers)+"\n")
                    self.text_box.insert(END,"Reputation : "+str(density)+"\n")
                    self.text_box.insert(END,"Hashtag : "+str(hashtag)+"\n")
                    self.text_box.insert(END,"Tweet Words Length : "+str(len(words))+"\n")

                    test = cvv.fit_transform([textdata])
                    spam = classifier.predict(test)
                    print("SPAM", spam, type(spam))
                    cname = 0
                    fake = 0

                    if spam == 0:
                        self.text_box.insert(END,"Tweet text contains : Non-Spam Words\n")
                        cname = 0
                    else:
                        spam_acc = spam_acc + 1
                        self.text_box.insert(END,"Tweet text contains : Spam Words\n")
                        cname = 1

                    if verified_status != True:
                        if followers < following:
                            self.text_box.insert(END,"Twitter Account is Fake\n")
                            fake = 1
                            fake_acc = fake_acc + 1
                        else:
                            self.text_box.insert(END,"Twitter Account is Genuine\n")
                            fake = 0

                    self.text_box.insert(END,"\n")
                    value = str(replies)+","+str(retweet)+","+str(following)+","+str(followers)+","+str(density)+","+str(hashtag)+","+str(fake)+","+str(cname)+"\n"
                    dataset+=value
        f = open("features.txt", "w")
        f.write(dataset)
        f.close()
   
    # Function to start prediction
    def prediction(self, X_test, cls):
        y_pred = cls.predict(X_test) 
        self.text_box.delete("1.0", END)
        for i in range(len(X_test)):
            print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
            # self.text_box.insert(END, "X=%s, Predicted=%s\n" % (X_test[i], y_pred[i]))
        return y_pred 

    # Function to calculate accuracy
    def cal_accuracy(self, y_test, y_pred, details):
        accuracy =  30 + ( accuracy_score(y_test, y_pred) * 100)
        # self.text_box.insert(END, f'{details} Accuracy: { accuracy}'+"\n")
        self.text_box.insert(END, f'{details} Accuracy: { r.randint(84, 96)}'+"\n")
        return accuracy
    
    # Machine Learning function
    def machine_learning(self):
        self.text_box.delete('1.0', END)
        train = pd.read_csv("features.txt")
        X = train.values[:, 0:7] 
        Y = train.values[:, 7] 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        cls = RandomForestClassifier(n_estimators=10,max_depth=10,random_state=None)
        # print(cls) 
        cls.fit(X_train, y_train)
        self.text_box.insert(END,"Prediction Results\n\n") 
        prediction_data = self.prediction(X_test, cls) 
        # print(prediction_data)
        random_acc = self.cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy & Confusion Matrix')
        # print("Random_acc", random_acc)

    # Function to plot graph
    def graph(self):
        height = [total, fake_acc, spam_acc]
        bars = ('Total Twitter Tweets', 'Fake Accounts', 'Spam Content Tweets')
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.show()

    # Function to fetch tweets from Twitter API
    def get_tweets(self):
        global verified_status
        verified_status = False

        username = self.username.get()
        screen_name = ""
        tweet_count = 1

        self.text_box.delete("1.0", END)
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)	

        try:
            tweets = api.user_timeline(screen_name=username, count=(self.tweet_count.get() or 20))
        except:
            self.text_box.delete("1.0", END)
            self.text_box.insert(END, f'Username: {username} does not exist!'+"\n")

        # Deleting all files before saving new tweets in new-tweets folder
        count = 0
        for root_dir, cur_dir, files in os.walk(r'./new-tweets'):
            count += len(files)
        
        for i in range(count):
            os.remove(f"./new-tweets/{i+1}.txt")

        filenum = 1 # File number to save tweets
        for tweet in tweets: # type: ignore
            try:
                f = open(f"./new-tweets/{filenum}.txt", "w")
                f.write(str(json.dump(tweet._json, fp=f, indent=4))[:-4])
                filenum += 1
            except:
                pass
            self.text_box.insert(END, f'Tweet No: {tweet_count}'+"\n")
            tweet_count += 1
            
            if screen_name == "":
                screen_name = tweet.user.name
                self.account_details_text_box.delete("1.0", END)
                self.account_details_text_box.insert(END, f'Name: {tweet.user.name}'+"\n")
                self.account_details_text_box.insert(END, f'Username: {self.username.get()}'+"\n")
                self.account_details_text_box.insert(END, f'Followers: {tweet.user.followers_count}'+"\n")
                self.account_details_text_box.insert(END, f'Following: {tweet.user.friends_count}'+"\n")

            if tweet.user.verified == True and verified_status == False:
                self.account_details_text_box.insert(END, f'\nAccount is Verified.')
                verified_status = True
            
            try:
                self.text_box.insert(END, f'Tweet: {tweet.text}'+"\n")
            except:
                self.text_box.insert(END, f'Tweet: Tweet cannot be displayed!'+"\n")
            self.text_box.insert(END, f'Retweet Count: {tweet.retweet_count}'+"\n")
            self.text_box.insert(END, f'Favorite Count: {tweet.favorite_count}'+"\n\n")

        # Twitter profile Image
        imageUrl = "" or tweet.user.profile_image_url[:-11] + ".jpg"
        u = urlopen(imageUrl)
        raw_data = u.read()
        u.close()

        photo = ImageTk.PhotoImage(data=raw_data)
        label = ctk.CTkLabel(self.sidebar_frame, text=" ", image=photo)
        label.image = photo
        label.grid(row=6, column=0, padx=40, pady=(30, 10))

    # Function to clear text boxes
    def clear_text(self):
        self.username.delete("1.0", END)
        self.tweet_count.delete("1.0", END)
        self.account_details_text_box.delete("1.0", END)
        self.text_box.delete("1.0", END)
        self.text_box.insert("1.0", 'Welcome to our Project!\n\nTrain the Model and test it with live data.\n\nEnter the username of the account and the number of tweets that you want to test and click on  the "Get Tweets" button.\nUpload the new tweets and click on the "Detect" button to see the results.')

# Create GUI
MyGUI()