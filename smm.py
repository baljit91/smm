import csv
import tweepy
import os.path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pylab
import time


DATA_PATH = os.path.join(os.getcwd(),"data")
stopwords = stopwords.words("english")
porter_stemmer = PorterStemmer()

hlist = ["whitehouse",
         #health
         "cdc_ehealth","CDCgov","healthcaregov","healthfinder",
         #fcc
         "FCC",
         "TheJusticeDept",
         #derug health
         "fda_drug_info","US_FDA","WomensHealthNIH","AIRNow","fda_drug_info",
         #education
         "edpartners","usedgov"
         #Defence
         ,"usarmy","ArmedwScience","usairforce","usnavy","uscoastguard",
         #weather
         "nws","NHC_Atlantic",
         #finance
         "libertystecon","GFOA","sbagov","USTreasury","FDICgov"]

def pre_processing(tweet):
    tweet = tweet.lower()
    res = ''.join([i if ord(i) < 128 and not i.isspace() else ' ' for i in tweet])
    res = res.strip()
    res_list = res.split(" ")
    res_list = [word.lower().strip() for word in res_list]
    res_list = [word.translate(None,string.punctuation) for word in res_list]
    res = " ".join([porter_stemmer.stem(word) for word in res_list if word not in stopwords and "http" not in word and word.strip() and word not in string.punctuation and word != "rt"])
    return res

def get_tweets(screen_name):

    API_KEY = "Xa6PPa9nTbNvfMrK7VC61DKRi"
    API_SECRET = "4oSOCpAP7zac4wb9aWzAvmNyY2gqzBIiBf7tsSFoFySlppTHej"

    auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
    alltweets = []
    while(len(alltweets)<1000):
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        new_tweets = api.user_timeline(screen_name=screen_name, count=200)
        alltweets.extend(new_tweets)
        if not new_tweets:
            break

    outtweets= [[tweet.user.followers_count,tweet.retweet_count,pre_processing(tweet.text.encode("utf-8"))] for tweet in alltweets]
    return outtweets

def write_to_csv(input_tweets,handler_name):
    save_path = DATA_PATH
    handler_name = "_".join([word.lower() for word in handler_name.split(" ")])
    completeName = os.path.join(save_path, handler_name + '.csv')
    df = pd.DataFrame(input_tweets,columns = ["followers","rt","text"])
    df.index.name = "Index"
    df.to_csv(os.path.join(save_path,completeName))
    print "The csv file is saved {0}".format(completeName)



def convert_to_df(dir_name):
    df = pd.DataFrame()
    hasher = {}
    for f in os.listdir(dir_name):
        if not f.startswith('.'):
            if not f in hasher.keys():
                df = pd.read_csv(os.path.join(DATA_PATH,f),encoding="utf-8")
                df.dropna()
                df["text"] = df["text"].values.astype(str)
                hasher[f] = df["text"]

    return hasher

    #     path =  os.path.join(DATA_PATH,f)
    #     print path
    #     df = pd.concat([df,pd.read_csv(path)])
    # return df

    # X = []
    # for i in range(0,len(all_tweets)):
    #     X.append(all_tweets[i][0])
    # df = pd.DataFrame(X)
    # return df

def bag_of_words(dataframe,file_location):
    tf = TfidfVectorizer(stop_words="english")
    transformed_input = tf.fit_transform(dataframe)
    transformed_input = transformed_input.toarray()
    return tf


def train_model(clusters=3):
    # LDA Topic MOdelling here.
    # Get Topics from unsupervised clusters
    Visualize
    model = KMeans(n_clusters=clusters)
    return model
#
# for handle in hlist:
#     tweet_list = get_tweets(handle)
#     write_to_csv(tweet_list,handle)


hasher = convert_to_df(DATA_PATH)
tweet_df = pd.concat(hasher.values())
# print tweet_df
tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,3))
res = tf_idf_vectorizer.fit_transform(tweet_df)
feature_names = tf_idf_vectorizer.get_feature_names()
kmeans_model = KMeans(n_clusters=8)
kmeans_model.fit(res)
clusters = kmeans_model.labels_.tolist()

tweet_df["cluster_id"] = clusters
print tweet_df["cluster_id"].value_counts()
centroids = kmeans_model.cluster_centers_
# print centroids
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

# plt.show()
