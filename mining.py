import pandas as pd
import nltk
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.cluster import KMeans
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
def split_csv():
	df=pd.read_csv("pkm.csv",names=['date','text'])
	dfnew=[]
	for i in range(19,27):
		dfnew.append(pd.DataFrame(columns=['date','text']))
	for i in range(df.shape[0]):
		s=df.iloc[i,0]
		if s[8:10] != '27':
			dfnew[int(s[8:10])-19].loc[i]=df.iloc[i]
	for i in range(19,27):
		dfnew[i-19].to_csv(f'pkm-{str(i)}.csv')
def ocluster():
	df=pd.read_csv("pkm-19-clean.csv")
	for i in range(20,27):
		df=df.append(pd.read_csv(f'pkm-{str(i)}-clean.csv'),ignore_index=True)
	text=np.array(df['text'])
	from sklearn.feature_extraction.text import CountVectorizer
	vec=CountVectorizer(stop_words='english')
	vec.fit(text)
	X=vec.transform(text)
	n=6
	k=KMeans(n_clusters=n)
	k.fit(X)
	df['clabel']=k.labels_
	sid = SentimentIntensityAnalyzer()
	for i in range(n):
		dfc=df[df['clabel']==i]
		sen=''
		for j in dfc['text']:
			sen+=j
		sen=sen.lower()
		toker=RegexpTokenizer(r'\w+')
		words=toker.tokenize(sen)
		stop_words = set(stopwords.words('english'))
		filtered_sentence = [w for w in words if not w in stop_words]
		fdist=FreqDist(filtered_sentence)
		sentive=[]
		for s in dfc['text']:
			ss=sid.polarity_scores(s)
			sentive.append(ss['compound'])
		sentive=np.array(sentive)
		print('counts:',dfc.shape[0],'pop words:',fdist.most_common(30),'Sentiment score:',np.average(sentive))
		print('------------------')
def scluster():
	for d in range(20,27):
		df=pd.read_csv(f'pkm-{str(d)}-clean.csv')
		text=np.array(df['text'])
		from sklearn.feature_extraction.text import CountVectorizer
		vec=CountVectorizer(stop_words='english')
		vec.fit(text)
		X=vec.transform(text)
		n=3
		k=KMeans(n_clusters=n)
		k.fit(X)
		df['clabel']=k.labels_
		sid = SentimentIntensityAnalyzer()
		for i in range(n):
			dfc=df[df['clabel']==i]
			sen=''
			for j in dfc['text']:
				sen+=j
			sen=sen.lower()
			toker=RegexpTokenizer(r'\w+')
			words=toker.tokenize(sen)
			stop_words = set(stopwords.words('english'))
			filtered_sentence = [w for w in words if not w in stop_words]
			fdist=FreqDist(filtered_sentence)
			sentive=[]
			for s in dfc['text']:
				ss=sid.polarity_scores(s)
				sentive.append(ss['compound'])
			sentive=np.array(sentive)
			print('date:',d,'counts:',dfc.shape[0],'pop words:',fdist.most_common(25),'Sentiment score:',np.average(sentive))
			print('-------------------------')
def preprossing():
	for i in range(19,27):
		df=pd.read_csv(f'pkm-{str(i)}.csv')
		df=df.drop_duplicates(['text'])
		print(df['text'].describe())
		df.to_csv(f'pkm-{str(i)}.csv')
def del_http_emoji(s):
	s1=re.sub(r'https?://.*? ','',s)
	s1=re.sub(r'https?://.*?$','',s1)
	s1=re.sub(r'b"','',s1)
	s1=re.sub(r"b'",'',s1)
	s1=re.sub(r'RT .*','',s1)
	s1=re.sub(r'\\x..','',s1)
	s1=re.sub(r'@.*? ','',s1)
	s1=re.sub(r'#.*? ','',s1)
	s1=re.sub(r'#.*?$','',s1)
	s1=re.sub(r'\\n','',s1)
	return s1
def del_deal():
	for i in range(19,27):
		df=pd.read_csv(f'pkm-{str(i)}.csv')
		df['text']=df['text'].apply(del_http_emoji)
		df=df[df['text']!='']
		df.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
		df=df.drop_duplicates(['text'])
		df.to_csv(f'pkm-{str(i)}-clean.csv')
def len_plot():
	df=pd.read_csv("pkm-19-clean.csv")
	for i in range(20,27):
		df=df.append(pd.read_csv(f'pkm-{str(i)}-clean.csv'),ignore_index=True)
	df['len']=df['text'].apply(len)
	df['len'].plot(bins=100,kind='hist')
	plt.xlabel('length')
	plt.show()
	'''
	print(df['len'].describe())
	print(df[df['len']==596]['text'].iloc[0])
	'''
def number_plot():
	a=[]
	for i in range(20,27):
		df=pd.read_csv(f'pkm-{str(i)}-clean.csv')
		a.append(df.shape[0])
	plt.plot(list(range(20,27)),a,marker='*')
	plt.xlabel("date")
	plt.ylabel('number of tweets')
	plt.show()
def sentiment_ana():
	y=[]
	for i in range(20,27):
		df=pd.read_csv(f'pkm-{str(i)}-clean.csv')
		sid = SentimentIntensityAnalyzer()
		sen=[]
		for s in df['text']:
			ss=sid.polarity_scores(s)
			sen.append(ss['compound'])
		sen=np.array(sen)
		y.append(np.average(sen))
	plt.plot(list(range(20,27)),y,marker='*')
	plt.xlabel("date")
	plt.ylabel('sentiment score')
	plt.show()
def pkmpopana():
	df=pd.read_csv("pkm-19-clean.csv")
	for i in range(20,27):
		df=df.append(pd.read_csv(f'pkm-{str(i)}-clean.csv'),ignore_index=True)
	sen=''
	for j in df['text']:
		sen+=j
	sen=sen.lower()
	toker=RegexpTokenizer(r'\w+')
	words=toker.tokenize(sen)
	stop_words = set(stopwords.words('english'))
	filtered_sentence = [w for w in words if not w in stop_words]
	fdist=FreqDist(filtered_sentence)
	pk=pd.read_csv('pokemon.csv')
	pk=pk[pk['id']<152]
	pkmname=list(pk['pokemon'])
	re={}
	for n in pkmname:
		if n in fdist.keys():
			re[n]=fdist[n]
	so=sorted(re.items(),key=lambda item:item[1],reverse = True)
	l,p=[],[]
	tar=so[0:2]
	for i in tar:
		l.append(i[1])
		p.append(i[0])
	plt.barh(list(range(len(tar))),width=l[::-1],align='center')
	plt.xlabel('count')
	plt.ylabel('name')
	plt.yticks(list(range(len(tar))),p[::-1])
	plt.show()
def pkmpopana_day():
	y=[]
	sid = SentimentIntensityAnalyzer()
	for i in range(20,27):
		df=pd.read_csv(f'pkm-{str(i)}-clean.csv')
		sen=''
		for j in df['text']:
			sen+=j
		sen=sen.lower()
		toker=RegexpTokenizer(r'\w+')
		words=toker.tokenize(sen)
		stop_words = set(stopwords.words('english'))
		filtered_sentence = [w for w in words if not w in stop_words]
		fdist=FreqDist(filtered_sentence)
		pkmname=['eevee','pikachu','vulpix','chansey','mewtwo']
		re={}
		for n in pkmname:
			if n in fdist.keys():
				#re[n]=fdist[n]
				senti=[]
				for s in df['text']:
					if n in s:
						ss=sid.polarity_scores(s)
						senti.append(ss['compound'])
				senti=np.array(senti)				
				re[n]=np.average(senti)
		y.append(re)
	c=['r','gray','purple','black','g']
	m=['*','.','^','x','P']
	for i in range(len(pkmname)):
		ys=[]
		for j in y:
			ys.append(j[pkmname[i]])
		plt.plot(list(range(20,27)),ys,label=pkmname[i],color=c[i],marker=m[i])
	plt.xlabel('date')
	plt.ylabel('sentiment score')
	plt.legend()
	plt.show()
def exp():
	pk=pd.read_csv('pokemon.csv')
	pk=pk[pk['id']<152]
	print(list(pk['pokemon']))
def extrem_sen_ex():
	df=pd.read_csv("pkm-19-clean.csv")
	for i in range(20,27):
		df=df.append(pd.read_csv(f'pkm-{str(i)}-clean.csv'),ignore_index=True)
	sid = SentimentIntensityAnalyzer()
	for s in df['text']:
		ss=sid.polarity_scores(s)
		if ss['compound']<-0.93:
			print(ss['compound'],s)
def mewtwo_lowsentimentextract():
	df=pd.read_csv(f'pkm-23-clean.csv')
	sid = SentimentIntensityAnalyzer()
	for s in df['text']:
		if 'mewtwo' in s:
			ss=sid.polarity_scores(s)
			if ss['compound'] < -0.2:
				print(ss['compound'],s)
def sentiment_ana_mul():
	y=[[],[],[]]
	for i in range(20,27):
		df=pd.read_csv(f'pkm-{str(i)}-clean.csv')
		sid = SentimentIntensityAnalyzer()
		sen=[[],[],[]]
		for s in df['text']:
			ss=sid.polarity_scores(s)
			sen[0].append(ss['pos'])
			sen[1].append(ss['neu'])
			sen[2].append(ss['neg'])
		sen[0]=np.array(sen[0])
		sen[1]=np.array(sen[1])
		sen[2]=np.array(sen[2])
		y[0].append(np.average(sen[0]))
		y[1].append(np.average(sen[1]))
		y[2].append(np.average(sen[2]))
	plt.plot(list(range(20,27)),y[0],marker='*',color='black',label='pos')
	plt.plot(list(range(20,27)),y[1],marker='.',color='r',label='neu')
	plt.plot(list(range(20,27)),y[2],marker='P',color='g',label='neg')
	plt.xlabel("date")
	plt.ylabel('sentiment score')
	plt.legend()
	plt.show()
def sentiment_ana_exwords():
	sen=''
	sid = SentimentIntensityAnalyzer()
	df=pd.read_csv("pkm-19-clean.csv")
	for i in range(20,27):
		df=df.append(pd.read_csv(f'pkm-{str(i)}-clean.csv'),ignore_index=True)
	for i in df['text']:
		ss=sid.polarity_scores(i)
		if ss['compound']>-0.05 and ss['compound']<0.05:
			sen+=i
	sen=sen.lower()
	toker=RegexpTokenizer(r'\w+')
	words=toker.tokenize(sen)
	stop_words = set(stopwords.words('english'))
	filtered_sentence = [w for w in words if not w in stop_words]
	fdist=FreqDist(filtered_sentence)
	print(fdist.most_common(50))
	
sentiment_ana_exwords()
