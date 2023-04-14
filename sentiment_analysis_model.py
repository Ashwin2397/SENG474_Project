from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

class SentimentAnalysisModel:
    def __init__(self, data):
        self.model = MultinomialNB()
        self.data = data
        self.train_test_data = None

        self.vectorizer: CountVectorizer = self.vectorizer()

    def preprocess_data(self, is_balance_data=True):
        # Drop neutral sentiment
        self.data = self.data[self.data.Sentiment != "neutral"] 
        
        if is_balance_data:
            self.balance_data()

        return self
    
    def balance_data(self):
        # Undersampling to balance the data
        _, neg_count = self.data['Sentiment'].value_counts()
        pos_data = self.data[self.data['Sentiment'] == 'positive']
        neg_data = self.data[self.data['Sentiment'] == 'negative']

        pos_under = pos_data.sample(neg_count)
        self.data = pd.concat([pos_under, neg_data], axis=0)

    def train_model(self):
        text_counts = self.vectorizer.fit_transform(self.data['Sentence'])

        #Splitting the data into training and testing
        self.train_test_data = train_test_split(text_counts, self.data['Sentiment'], test_size = 0.20, random_state = 6, stratify = self.data['Sentiment']) # VS stratify = data['Sentiment']
        X_train, _, Y_train, _ = self.train_test_data

        self.model.fit(X_train, Y_train)

        return self

    def vectorizer(self):
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
        return cv
        
    def predict(self, sentence):
        vectorized_sentence = self.vectorizer.transform([sentence])
        return self.model.predict(vectorized_sentence)
    