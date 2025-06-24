# preprocess.py
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
tokenizer = ToktokTokenizer()

url_pattern = re.compile(r'http\S+|www\S+|https\S+')
mention_pattern = re.compile(r'@\w+')
special_char_pattern = re.compile(r'[^\w\s]')

def preprocess_and_stem(tweet):
    tweet = str(tweet).lower()
    tweet = url_pattern.sub('', tweet)
    tweet = mention_pattern.sub('', tweet)
    tweet = special_char_pattern.sub('', tweet)
    tokens = tokenizer.tokenize(tweet)
    processed_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(processed_tokens)

