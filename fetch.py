import tweepy
import csv
csvFile = open('pkm.csv', 'w')
csvWriter = csv.writer(csvFile)
apiKey="8pMMW7G8hMvhBoEr7ETU11ORY"
apiSecretKey="x0XZFDCTsoDOUs74MRmLta0rBZCB9e8aJKB5njnenOGZDzTJ3x"
accessToken="992672020000321536-YvA56tOigbmTEaZTvWYoKF90PCaBJQ6"
accessTokenSecret="se0YyCDjfAH1mq8NKIHhQPt5guRtmGC2BQboVcJRRSCwR"
auth=tweepy.OAuthHandler(apiKey,apiSecretKey)
auth.set_access_token(accessToken,accessTokenSecret)
api=tweepy.API(auth,wait_on_rate_limit=True)
for t in tweepy.Cursor(api.search,q='#PokemonLetsGo',\
	lang='en',since='2018-11-11',tweet_mode='extended').items():
	#print(t.created_at,t.full_text)
	csvWriter.writerow([t.created_at, t.full_text.encode('utf-8')])