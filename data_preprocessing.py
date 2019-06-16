import pandas as pd
from sklearn import preprocessing

class data_preprocessing:
    def process(name):
        df = pd.read_csv(name)
        df = df.drop(columns=['Div'])
        df = df.drop(columns=['Date'])
        df = df.drop(columns=['Referee'])
        df = df.replace({'H': '1', 'A': '2', 'D':'0'})
        return df