import csv
import tweepy
import os.path
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pylab
import time
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA


DATA_PATH = os.path.join(os.getcwd(),"data")
stopwords = stopwords.words("english")
porter_stemmer = SnowballStemmer("english", ignore_stopwords=True)

hlist = ["whitehouse","presssec","realdonaldtrump"
         #health
         "cdc_ehealth","CDCgov","healthcaregov","womenshealth",
         #Home Affairs
         "TheJusticeDept","StateDept","DeptofDefense"
         #education
         "edpartners","usedgov","arneduncan","NAEYC"
         #Defence
         ,"usarmy","usairforce","usnavy",
         #weather
         "nws","NOAA","NWSWPC"
         #finance
         "libertystecon","GFOA","sbagov","USTreasury"]

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



#
# for handle in hlist:
#     tweet_list = get_tweets(handle)
#     write_to_csv(tweet_list,handle)


hasher = convert_to_df(DATA_PATH)
tweet_df = pd.concat(hasher.values())
# print tweet_df
print "Modelll"
tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,3),smooth_idf=True)
res = tf_idf_vectorizer.fit_transform(tweet_df)
tf_feature_names = tf_idf_vectorizer.get_feature_names()
no_of_clusters= 8
kmeans_model = KMeans(n_clusters=8,max_iter=5)
kmeans_model.fit(res)
clusters = kmeans_model.labels_.tolist()
tweet_df["cluster_id"] = clusters
print tweet_df["cluster_id"].value_counts()
print "Top words from KMeans"

order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
for i in range(8):
    print("Cluster {} : Words :".format(i))
    for ind in order_centroids[i, :10]:
        print(' %s' % tf_feature_names[ind])

dim = 2
#Setting dimensions to populate the graph.
pca= PCA(n_components=2).fit(res.todense())
data2D = pca.transform(res)
plt.scatter(data2D[:,0], data2D[:,1], c=clusters)
plt.show()





# no_topics = 4


# lda = LatentDirichletAllocation(max_iter=15, learning_method='online', learning_offset=50.,random_state=10).fit(res)
#
#
#
# def display_topics(model, feature_names, no_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         print ("Topic %d:" % (topic_idx))
#         print (" ".join([feature_names[i]
#                         for i in topic.argsort()[:-no_top_words - 1:-1]]))
#
# no_top_words = 10
# display_topics(lda, tf_feature_names, no_top_words)


# from sklearn.cluster import KMeans
#
# plt.figure(figsize=(40, 50))
# plt.scatter(res[:,0], res[:,1])
# print centroids
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)

# plt.show()
