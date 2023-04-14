# Description
The aim of this project is to create a model that predicts the change in price given the news polarity score at that point in time.

# Navigation
- `report.ipynb`: contains all phases of this project rolled into one file.
- `./sentiment-analysis/bow_binary.ipynb`: contains details surrrounding the training of the sentiment analysis model. 
- `./data/data_process.ipynb`: an attempt at preprocessing tweets and reddit data and analysing emoticons.
- `./sentiment-analysis/`: contains varying attempts at the sentiment analysis training to determine the optimal model of choice.
- `df.csv`: the resulting dataframe used to train the price prediction model
- `get_price_data.py`: a service object tasked with pulling from the Yahoo Finance API and injecting the predicted sentiment scores to the output dataframe.
- `sentiment_analysis_model.py`: a service object created to reflect the sentiment analysis model created in `bow_bin.ipynb`
- `utils.py`: a module created for simple utility functions used in `report.ipynb`
