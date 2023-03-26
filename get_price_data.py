import datetime as dt
import yfinance as yf
from random import random
import pandas as pd

class PriceDataRetrieverAndPreprocessor:
    """
    Retrieves price data from Yahoo Finance and preprocesses it to an acceptable format that may be used for training.
    The exhaustive list of columns that may be included are as follows:
    ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Difference', 'Price polarity over interval', 'Start time', 'End time', 'News polarity at start time']

    By default, the following columns are dropped:
    ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Difference']
    
    Therefore implying that the following columns are included:
    ['Start time', 'End time', 'News polarity at start time', 'Price polarity over interval']

    The 'Price polarity over interval' column is essentially the label that we are trying to predict.
    """
    def __init__(self, ticker, start, end, 
                 polarity_score_df, 
                 columns_to_drop=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 
                 interval='60m'):
        self.interval = interval
        self.ticker = ticker
        self.start = start
        self.end = end

        self.polarity_score_df = polarity_score_df

        self.df = None

        self.columns_to_drop = columns_to_drop

    def retrieve(self):
        """
        Get price data from Yahoo Finance
        """
        # get price data from Yahoo Finance
        ticker = yf.Ticker(self.ticker)
        self.df = ticker.history(
            start=self.start,
            end=self.end,
            interval=self.interval
        ).reset_index()

        return self

    def preprocess(self):
        """
        Preprocess data to an acceptable format that may be used for training
        """
        self.inject_price_polarity()
        self.inject_start_end_time()
        self.inject_polarity_score()
        
        self.drop_columns()

        return self.df

    def inject_price_polarity(self):
        """
        Get price polarity, which is determined by the difference between the open and close price
        """

        self.df['Difference'] = self.df.apply(lambda row: row['Close'] - row['Open'], axis=1)
        # 0 if the difference is negative and 1 if the difference is positives
        # TODO: Include one more label for when the price does not change
        self.df['Price polarity over interval'] = self.df['Difference'].apply(lambda x: 1 if x > 0 else 0)

        self.columns_to_drop.append('Difference')
        
    def inject_start_end_time(self):
        interval_string_to_hours = { f'{i}m':i/60 for i in range(0, 65, 5)}
        self.df = self.df.rename(columns={'Datetime': 'Start time'})
        self.df['End time'] = self.df.apply(lambda row: row['Start time'] + dt.timedelta(hours=interval_string_to_hours[self.interval]), axis=1)

        self.convert_timestamp()

    def convert_timestamp(self):
        self.df['Start time'] = pd.to_numeric(pd.to_datetime(self.df['Start time']))
        self.df['End time'] = pd.to_numeric(pd.to_datetime(self.df['End time']))


    def inject_polarity_score(self):
        # TODO: This has to be changed to actually grab the polarity score from the self.polarity_score_df
        self.df['News polarity at start time'] = self.df.apply(lambda _: random(), axis=1)

    def drop_columns(self):
        self.df = self.df.drop(columns=self.columns_to_drop)


