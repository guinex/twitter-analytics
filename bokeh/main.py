from random import random

from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import Select,TextInput
from bokeh.layouts import widgetbox
from bokeh.io import reset_output
import sys
import time
from twitter.api import TwitterHTTPError
from urllib.error import URLError
from http.client import BadStatusLine
import pandas as pd
import json 

# To make it more readable, lets store
# the OAuth credentials in strings first.
with open('twitterData.txt','r') as f:
    api_auth = json.load(f)
from twitter import *



CONSUMER_KEY = api_auth['API_keys']
CONSUMER_SECRET = api_auth['API_secret_key']
OAUTH_TOKEN = api_auth['Access_token']
OAUTH_TOKEN_SECRET = api_auth['Access_token_secret']

key_secret = '{}:{}'.format(CONSUMER_SECRET, OAUTH_TOKEN_SECRET).encode('ascii')
b64_encoded_key = base64.b64encode(key_secret)
b64_encoded_key = b64_encoded_key.decode('ascii')

#fetching data
cort_df = pd.read_csv(r'../mining/followers_cortez_analysis.csv')
cren_df = pd.read_csv(r'../mining/followers_crenshaw_analysis.csv')


dataset_cortez=cort_df.drop(columns=['ID'])
dataset_cren=cren_df.drop(columns=['ID'])

dataset_cortez=dataset_cortez.loc[(dataset_cortez!=0).any(1)]
dataset_cren=dataset_cren.loc[(dataset_cren!=0).any(1)]

dataset_cortez['main'] = 'cortez'
dataset_cren['main'] = 'crenshaw'


dataset_total = dataset_cortez.append(dataset_cren)
dataset_total = dataset_total.sample(frac=1).reset_index(drop=True)
X = dataset_total.iloc[:,0:21] #df1 = df.iloc[:,0:2]
y = dataset_total.iloc[:, [21]]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)
#end
#start twitter requests


# Loading my authentication tokens
#with open('auth_dict','r') as f:
#    api_auth = json.load(f)


# Then, we store the OAuth object in "auth"
auth = OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)
# Notice that there are four tokens - you need to create these in the
# Twitter Apps dashboard after you have created your own "app".

# We now create the twitter search object.
t = Twitter(auth=auth)


def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
        if wait_period > 3600: # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e
        if e.e.code == 401:
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429:
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60*15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered %i Error. Retrying in %i seconds' % (e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0
    while True:
        try:
            return twitter_api_func(*args, **kw)
        except TwitterHTTPError as e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            print >> sys.stderr, "BadStatusLine encountered. Continuing."
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise

# This will let us create new partial
# functions with arguments set to 
# certain values.
from functools import partial

# This was maxint.
# There is no longer a maxint (in Python 3)
from sys import maxsize


def get_friends_followers_ids(twitter_api, screen_name=None, user_id=None,
                                friends_limit=maxsize, followers_limit=maxsize):
    # Must have either screen_name or user_id (logical xor)
    assert (screen_name != None) != (user_id != None), \
    "Must have screen_name or user_id, but not both"
    
    # You can also do this with a function closure.
    get_friends_ids = partial(make_twitter_request, twitter_api.friends.ids,
                                count=15000)
   # get_followers_ids = partial(make_twitter_request, twitter_api.followers.ids,
    #                            count=15000)
    friends_ids= []
    for twitter_api_func, limit, ids, label in [
            [get_friends_ids, friends_limit, friends_ids, "friends"]
            ]:
        #LOOK HERE! This little line is important.
        if limit == 0: continue
        cursor = -1
        while cursor != 0:
            # Use make_twitter_request via the partially bound callable...
            if screen_name:
                response = twitter_api_func(screen_name=screen_name, cursor=cursor)
            else: # user_id
                response = twitter_api_func(user_id=user_id, cursor=cursor)
            if response is not None:
                ids += response['ids']
                cursor = response['next_cursor']
            print('Fetched {0} total {1} ids for {2}'.format(len(ids),
                    label, (user_id or screen_name), file=sys.stderr))
            if len(ids) >= limit or response is None:
                break
    # Do something useful with the IDs, like store them to disk...
    return friends_ids[:friends_limit]


following_oliver = following_meyers = following_colbert=following_sambee = following_hannity = following_tucker=following_maddow = following_tapper = following_cooper=following_lindsay= following_cruz = following_mitch=following_ryan=following_bernie=following_warren= following_kamala = following_beto=following_dogs=following_rowling=following_james=following_musk=0
import tweepy
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth)

#john oliver,seth meyers, colbert,sam bee
user = api.get_user(screen_name = 'iamjohnoliver')
user_oliver=user.id #oliver
user = api.get_user(screen_name = 'LastWeekTonight')
user_oliver_show=user.id #oliver_show

user = api.get_user(screen_name = 'StephenAtHome')
user_colbert=user.id #cobert
user = api.get_user(screen_name = 'colbertlateshow')
user_colbert_show=user.id #cobert_show

user = api.get_user(screen_name = 'sethmeyers')
user_meyers=user.id #meyers
user = api.get_user(screen_name = 'LateNightSeth')
user_meyers_show =user.id #meyers_show

user = api.get_user(screen_name = 'iamsambee')
user_sambee=user.id #sambee
user = api.get_user(screen_name = 'FullFrontalSamB')
user_sambee_show=user.id #sambee_show
#news-hannity, tucker carlson, rachel maddow, tapper, cooper
user = api.get_user(screen_name = 'jaketapper')
user_tapper=user.id 
user = api.get_user(screen_name = 'TheLeadCNN')
user_tapper_show=user.id 

user = api.get_user(screen_name = 'seanhannity')
user_hannity=user.id 
user = api.get_user(screen_name = 'TuckerCarlson')
user_carlson=user.id 
user = api.get_user(screen_name = 'maddow')
user_maddow=user.id

user = api.get_user(screen_name = 'andersoncooper')
user_cooper=user.id 
user = api.get_user(screen_name = 'AC360')
user_cooper_show=user.id 
#lindsay graham,ted cruz, leader mitch mcconel, paul ryan
user = api.get_user(screen_name = 'LindseyGrahamSC')
user_graham=user.id 
user = api.get_user(screen_name = 'tedcruz')
user_cruz=user.id 
user = api.get_user(screen_name = 'senatemajldr')
user_mitch=user.id 
user = api.get_user(screen_name = 'SpeakerRyan')
user_ryan=user.id 
#bernie, elizabeth warren,kamala harris, beto rourke
user = api.get_user(screen_name = 'BernieSanders')
user_bernie=user.id 
user = api.get_user(screen_name = 'SenWarren')
user_warren=user.id 
user = api.get_user(screen_name = 'KamalaHarris')
user_kamala=user.id 
user = api.get_user(screen_name = 'BetoORourke')
user_beto=user.id 
# weratedogs, j k rowling, lebron james, elon musk
user = api.get_user(screen_name = 'dog_rates')
user_doggo=user.id 
user = api.get_user(screen_name = 'jk_rowling')
user_rowling=user.id 
user = api.get_user(screen_name = 'KingJames')
user_james=user.id 
user = api.get_user(screen_name = 'elonmusk')
user_elon=user.id 

#end twitter requests



#create one more plot
#source = ColumnDataSource(data=dict(x=x, y=y))
plot = figure(tools="reset",x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
plot.border_fill_color = 'blue'
plot.background_fill_color = 'blue'
plot.outline_line_color = None
plot.grid.grid_line_color = None
plot = figure(plot_width=400, plot_height=400)
#plot.vbar(x=[1,3], width=0.5, bottom=0,top=[x,y], color=["blue","red"],source=source)

def plotgraph(attr, old, new):
    
    count_cortez = len(cort_df.loc[cort_df[new] == 1])
    count_cren = len(cren_df.loc[cren_df[new] == 1])
    
    plot.vbar(x=[1,3], width=0.5, bottom=0,top=[count_cortez, count_cren], color=["blue","red"])
    
# add a text renderer to our plot (no data yet)

select = Select(title="Option:", value=" ", options=["",'oliver','meyers','colbert','sambee','hannity','carlson','maddow','tapper','cooper','lindsay','cruz','mitch','paulryan','berie','warren','kamala','beto'])
select.on_change('value', plotgraph)

#put a text field
def my_text_input_handler(attr, old, new):
    user = api.get_user(screen_name = new)
    user_id=user.id
    for i in [user_id]:
        friends_ids = get_friends_followers_ids(t,
                                    user_id=int(i))
        if user_graham in friends_ids :
            following_lindsay = 1
        else:
            following_lindsay=0
        if user_cruz in friends_ids :
            following_cruz = 1
        else:
            following_cruz=0
        if user_mitch in friends_ids :
            following_mitch = 1
        else:
            following_mitch=0
        if user_ryan in friends_ids :
            following_ryan = 1
        else:
            following_ryan=0
        if user_colbert in friends_ids or user_colbert_show in friends_ids :
            following_colbert = 1
        else:
            following_colbert=0

        if user_oliver in friends_ids or user_oliver_show in friends_ids :
            following_oliver = 1
        else:
            following_oliver=0

        if user_meyers in friends_ids or user_meyers_show in friends_ids:
            following_meyers = 1
        else:
            following_meyers=0
        if user_sambee in friends_ids or user_sambee_show in friends_ids:
            following_sambee = 1
        else:
            following_sambee=0

        if user_hannity in friends_ids:
            following_hannity=1
        else:
            following_hannity = 0
        if user_carlson in friends_ids :
            following_tucker = 1
        else:
            following_tucker =0
        if user_maddow in friends_ids :
            following_maddow = 1
        else:
            following_maddow=0
        if user_tapper in friends_ids or user_tapper_show in friends_ids :
            following_tapper = 1
        else:
            following_tapper=0
        if user_cooper in friends_ids or user_cooper_show in friends_ids:
            following_cooper=1
        else:
            following_cooper = 0

        if user_bernie in friends_ids:
            following_bernie=1
        else:
            following_bernie = 0
        if user_warren in friends_ids :
            following_warren = 1
        else:
            following_warren =0
        if user_kamala in friends_ids :
            following_kamala = 1
        else:
            following_kamala=0
        if user_beto in friends_ids :
            following_beto = 1
        else:
            following_beto=0

        if user_doggo in friends_ids:
            following_dogs=1
        else:
            following_dogs = 0
        if user_rowling in friends_ids :
            following_rowling = 1
        else:
            following_rowling =0
        if user_james in friends_ids :
            following_james = 1
        else:
            following_james=0
        if user_elon in friends_ids :
            following_musk = 1
        else:
            following_musk=0
    #df = pd.DataFrame(columns=['ID','oliver','meyers','colbert','sambee','hannity','carlson','maddow','tapper','cooper','lindsay','cruz','mitch','paulryan','berie','warren','kamala','beto'])    
        df = pd.DataFrame([[i, following_oliver,following_meyers,following_colbert,following_sambee,following_hannity,following_tucker,following_maddow,following_tapper,following_cooper,following_lindsay,following_cruz,following_mitch,following_ryan,following_bernie,following_warren,following_kamala,following_beto,following_dogs,following_rowling,following_james,following_musk]], columns=['ID','oliver','meyers','colbert','sambee','hannity','carlson','maddow','tapper','cooper','lindsay','cruz','mitch','paulryan','berie','warren','kamala','beto','ratedogs','rowling','james','elonmusk'])
    df=df.drop(columns=['ID'])
    y_pred = classifier.predict(df)
    print("Previous label: " + old)
    print("Updated label: " + new)
    print(y_pred)
    text_input.title=y_pred[0]

text_input = TextInput(value="default", title="Label:")
text_input.on_change("value", my_text_input_handler)
# put the button and plot in a layout and add to the document

curdoc().add_root(column(select,plot,text_input))
