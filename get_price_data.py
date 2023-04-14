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
        self.convert_timestamp()
        
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

    def convert_timestamp(self):
        self.df['Start time'] = pd.to_numeric(pd.to_datetime(self.df['Start time']))
        self.df['End time'] = pd.to_numeric(pd.to_datetime(self.df['End time']))

    def inject_polarity_score(self):
        # Loop through each row in the dataframe
        for index, row in self.df.iterrows():
            # Loop through each row in the polarity score dataframe
            for polarity_index, polarity_row in self.polarity_score_df.iterrows():
                # If the polarity score is within the range of the start and end time, inject the polarity score into the dataframe
                if self.time_within_range(polarity_row['created_at_est'], row['Start time'], row['End time']):
                    self.df.at[index, 'News polarity at start time'] = polarity_row['polarity']
                    # break

        # Loop through each row in the dataframe and drop the rows that do not have a polarity score
        for index, row in self.df.iterrows():
            if pd.isna(row['News polarity at start time']):
                self.df = self.df.drop(index=index)

    def time_within_range(self, time, start, end):
        '''
        Determines if the `time` is within the range of `start` and `end`
        '''
        return pd.to_datetime(time) >= pd.to_datetime(start) and pd.to_datetime(time) <= pd.to_datetime(end)

    def drop_columns(self):
        self.df = self.df.drop(columns=self.columns_to_drop)


