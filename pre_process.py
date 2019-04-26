tweet_id = []
label = []
tweet_text = []
df = pd.read_csv('twitter-2013dev-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2013test-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())


df = pd.read_csv('twitter-2013train-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2014sarcasm-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2014test-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2015test-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2015train-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2016dev-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2016devtest-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())

df = pd.read_csv('twitter-2016test-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.index.tolist())
label.extend(df.tweet_id.tolist())
tweet_text.extend(df.label.tolist())

df = pd.read_csv('twitter-2016train-A.tsv',sep = '\t',names = ['tweet_id','label','tweet_text'])
tweet_id.extend(df.tweet_id.tolist())
label.extend(df.label.tolist())
tweet_text.extend(df.tweet_text.tolist())


df = pd.DataFrame()
df['tweet_id'] = tweet_id
df['label'] = label
df['tweet_text'] = tweet_text

df.to_csv('tweet_data.csv',index = False)