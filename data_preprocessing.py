import pandas as pd


class data_preprocessing:
    def process(name):
        df = pd.read_csv(name)
        # TODO without the following conversation the classifier fails. why?
        df = df.replace({'H': '1', 'A': '2', 'D': '0'})

        # we drop most of the columns as they contain
        # live data of the game
        df = df[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR']]
        return df
