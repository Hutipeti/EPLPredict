import csv
import pandas as pd
import data_preprocessing as dp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#read the csv into pandas dataframe and do some data cleaning on it
df =  dp.data_preprocessing.process('season-1819_csv.csv')

#separate the target feature from the rest
X = df.drop(columns=['FTR'])
y = df['FTR'].astype(int)

#split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y)

#create the neural network and train it
mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20),max_iter=2000)
mlp.fit(X_train,y_train)
# run preedictions on the test data and visualize the results
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))

#iterate over the confusion matrix elements and expose all wrongly predicted datarow in a csv file
#named wrongly_classified_testdata.csv

# TODO get back the original team names!

misclassified_samples = X_test[y_test != predictions]
output_mis = misclassified_samples.to_csv(index=False)

f = open("prediction_outcome/wrongly_classified_testdata.csv","w")
f.write(output_mis)
f.close()
print(misclassified_samples)