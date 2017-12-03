import csv
import tweepy
import os.path


def preprocessing(tweet):
    


#get tweets from twitter
def get_all_tweets(screen_name):

    API_KEY = "Xa6PPa9nTbNvfMrK7VC61DKRi"
    API_SECRET = "4oSOCpAP7zac4wb9aWzAvmNyY2gqzBIiBf7tsSFoFySlppTHej"

    auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    alltweets = []
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    alltweets.extend(new_tweets)
    
    outtweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]
    return outtweets
        
#write twwets into csv file
def write_into_file(input_tweets,handler_name):
    save_path = '/Users/Singh/Data'
    completeName = os.path.join(save_path, handler_name + '.csv') 
    with open(completeName, 'w') as f:
        writer = csv.writer(f)
        #writer.writerow(["text & classifier"])
        writer.writerows(input_tweets)
        f.close()

        #white house

hlist = ["whitehouse","jesseclee44",
         #health
         "cdc_ehealth","CDCgov","healthcaregov","healthfinder",
         #fcc
         "FCC",
         "TheJusticeDept",
         #derug health
         "fda_drug_info","US_FDA","WomensHealthNIH","AIRNow","fda_drug_info",
         #education
         "edpartners","usedgov","teachgov"
         #Defence
         ,"usarmy","ArmedwScience","usairforce","usnavy","uscoastguard",
         #weather
         "usnwsgov","NHC_Atlantic",
         #finance
         "libertystecon","GFOA","NYFed_news","sbagov","USTreasury","FDICgov"]

for hand in hlist:
    handler_name = hand
    handler_tweets = get_all_tweets(handler_name)
    #write the csv
    write_into_file(handler_tweets,handler_name)
    print(hand)







    
#This depends on the results we are obtaining below.More like a feedback system


#read from csv files
all_tweets = []
import csv
save_path = '/Users/Singh/Data'

for handler_name in hlist:
    completeName = os.path.join(save_path, handler_name + '.csv')
    with open(completeName) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            all_tweets.append(row)
    print(handler_name)
            #print(row)
print(len(all_tweets))








from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

print(all_tweets[0][0])

X = []
Y = []
for i in range(0,len(all_tweets)):
    X.append(all_tweets[i][0])
    #Y.append(all_tweets[i][0][1])


df = pd.DataFrame(X)
X = df[0]

#df = pd.DataFrame(Y)
#Y = df[0]

tf = TfidfVectorizer(stop_words="english",min_df = 35)
transforemd_input = tf.fit_transform(X)
transforemd_input = transforemd_input.toarray()









from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

true_k = 8
#model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model = KMeans(init='k-means++', max_iter=100, n_init=1)
model.fit(transforemd_input)










from sklearn.feature_extraction.text import TfidfVectorizer

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = tf.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i,)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind],)
    print()

