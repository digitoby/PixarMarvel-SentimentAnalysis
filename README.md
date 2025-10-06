# Disney Sentiment Analysis - Pixar vs. Marvel Phase 5
This project analyzes over 28,000 tweets and Reddit posts to compare audience sentiment toward Pixar films and Marvel Phase 5 releases. Using data collected through the X API and Reddit API, I built Boolean queries to capture relevant discussions and performed natural language processing (NLP) for sentiment classification.

Due to strict API limitations for free users like me, however, I decided to narrow the focus of this project to produce better results. Therefore, I only made requests related to Inside Out 2 and Deadpool & Wolverine. Both films were released in the Summer of 2024 and did outstandingly well in the global box office, making them perfect for a comparison.

I leveraged NLTK and TextBlob to conduct the sentiment analysis and gave special attention when handling multilingual text to ensure accurate sentiment detection. The processed data was then visualized in a Tableau dashboard, highlighting key insights such as:

* Overall sentiment toward Inside Out 2 and Deadpool & Wolverine

* Amount of online conversation produced by each movie

* Subreddit contribution to reveal which communities were the most active

The goal of this project was to demonstrate how data collection, NLP, and visualization can be combined to uncover meaningful insights into consumer conversations around major entertainment franchises.

## Gathering the data
First, I needed to scrape posts from social media platforms.

#### Twitter (X)
Twitter's API is called Tweepy which enforces strict pulling limits for users like me. I used this code to evenly distribute the number of pulls I was alotted to each movie:

```Python
bearer_token = "####"
client = tweepy.Client(bearer_token=bearer_token)'''

# make boolean queries
pixar_query = (
    '"Inside Out 2" OR "Inside Out II" OR "Inside Out Two"'
)
marvel_query = (
    '"Deadpool & Wolverine" OR "Deadpool and Wolverine" OR "Wolverine and Deadpool"'
)

# add lang filter inside the query
pixar_query += " lang:en"
marvel_query += " lang:en"

# fetch tweets function using free developer plan
'''def fetch_tweets_free(query, max_results=50):
    response = client.search_recent_tweets(
        query=query,
        tweet_fields=['id','text','author_id','created_at','public_metrics'],
        max_results=max_results,
    )
    tweets = response.data
    if tweets is None:
        return pd.DataFrame()  # return empty df if no tweets
    df = pd.DataFrame([{
        'id': t.id,
        'text': t.text,
        'author_id': t.author_id,
        'created_at': t.created_at.isoformat(),
        'retweets': t.public_metrics['retweet_count'],
        'likes': t.public_metrics['like_count'],
        'replies': t.public_metrics['reply_count'],
        'quotes': t.public_metrics['quote_count']
    } for t in tweets])
    return df'''
```
This function allowed me to make requests from Tweepy as long as I gave it the keywords. To make sure that we only pulled English tweets for best results, I added the "lang:en" parameter to the queries. I then called the function to create our dataset:

```Python
pixar_tweets = fetch_tweets_free(pixar_query)
marvel_tweets = fetch_tweets_free(marvel_query)

pixar_tweets.to_csv('pixar_tweets.csv')
marvel_tweets.to_csv('marvel_tweets.csv')
```
In the above code, I called the function to scrape tweets and create our dataset. I then saved them as CSVs to make sure we don't lose the data.

#### Reddit (PRAW)
PRAW is a powerful library for pulling from Reddit's API. 

```Python
movies = {
    "Inside Out 2": ["Inside Out 2", "Inside Out II", "Inside Out Two", "Inside Out"],
    "Deadpool and Wolverine": ["Deadpool and Wolverine", "Deadpool & Wolverine", "Wolverine and Deadpool", "Deadpool", "Wolverine"]
}

subreddits = ["movies", "boxoffice", "Pixar", "marvelstudios"]

all_posts = []

for movie, keywords in movies.items():
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.top(limit=1500):  # adjust limit as needed
            # Check if post matches any keyword
            combined_text = f"{post.title} {post.selftext}".lower()
            if any(k.lower() in combined_text for k in keywords):
                # Append the post itself
                all_posts.append({
                    "movie": movie,
                    "subreddit": sub,
                    "type": "post",  # differentiate posts from comments
                    "text": combined_text,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "url": post.url,
                    "comment_id": None
                })
                
                # Pull all comments for this post
                post.comments.replace_more(limit=50)
                for comment in post.comments.list():
                    all_posts.append({
                        "movie": movie,
                        "subreddit": sub,
                        "type": "comment",
                        "text": comment.body,
                        "score": comment.score,
                        "num_comments": post.num_comments,
                        "created_utc": comment.created_utc,
                        "url": post.url,
                        "comment_id": comment.id
                    })
```
As you can see, I used a for loop this time. Using this method, I was able to pull over 28k posts from Reddit regarding Inside Out 2 and Deadpool & Wolverine.

## Cleaning
Cleaning this dataset was complex. I decided the best way to create a useable dataset for Tableau analysis was to merge them all into one. I did so with the following code:

```Python
# delete unecessary columns, create column 'movie' and platform and concat so we have one df
pixar_tweets = pixar_tweets.drop(['Unnamed: 0'], axis=1)
marvel_tweets = marvel_tweets.drop(['Unnamed: 0'], axis=1)

# create a movie column for analysis flexibility
pixar_tweets['movie'] = 'Inside Out 2'
marvel_tweets['movie'] = 'Deadpool and Wolverine'

# concat!
twitter_data = pd.concat([pixar_tweets, marvel_tweets], ignore_index=True)

# create a source column for analysis flexibility
reddit_data['source'] = 'reddit'
twitter_data['source'] = 'twitter'

# change their dates so they're in the same format
reddit_data["created_utc"] = pd.to_datetime(reddit_data["created_utc"], unit="s")
twitter_data["created_at"] = pd.to_datetime(twitter_data["created_at"])

# get rid of timezone so they're exactly the same
twitter_data["created_at"] = twitter_data["created_at"].dt.tz_convert(None)

# CONCAT!!!
combined_data = pd.concat([twitter_data, reddit_data], ignore_index=True)
```

The cleaning also included column renaming, reordering, and ID synthesizing but I did not feel that it was imortant for this break down. To ensure a smooth merge, I standardized the format of 'time-posted' columns, created additional columns to differentiate posts by their platform and movie, and finally concatenated them for the final dataset.

Lastly, for best results from the sentiment analysis, I removed emojis, special characters, links, and mentions from the text data (column containing the actual content from posts).
```Python
import re
import emoji

# function to clean text
def clean_text(text):
    text = text.lower()
    
    # remove links
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # remove special characters except letters, numbers, spaces, sentence delimiters, and emojis
    cleaned_chars = []
    for char in text:
        if char.isalnum() or char.isspace() or char in '.!?':
            cleaned_chars.append(char)
        elif emoji.is_emoji(char):
            cleaned_chars.append(char)
    text = ''.join(cleaned_chars)
    
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# apply to the text column
combined_data['clean_text'] = combined_data['text'].apply(clean_text)

combined_data['clean_text']
```

## NLP
