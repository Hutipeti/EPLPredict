import pandas as pd
import data_preprocessing as dp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#read the csv into dandas dataframe and do some cleaning on it
df =  dp.data_preprocessing.process('season-1819_csv.csv')
#print(df)0
X = df.drop(columns=['FTR'])
y = df['FTR'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y)

#create the neural network and train it
mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20),max_iter=2000)
mlp.fit(X_train,y_train)
# run preedictions on the test data and visualize the results
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

