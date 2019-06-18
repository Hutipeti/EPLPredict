import data_preprocessing as dp
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import datetime


def run_experiment():
    # read the csv file into pandas dataframe and do some data cleaning on it
    df = dp.data_preprocessing.process('data/season-1819_csv.csv')

    # separate the target feature from the rest
    X = df.drop(columns=['FTR'])
    y = df['FTR'].astype(int)

    # split the data into train and test (default)
    # TODO implement cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # create the neural network and train it
    mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20), max_iter=2000)
    mlp.fit(X_train, y_train)

    # run preedictions on the test data and display some metrics
    predictions = mlp.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test,predictions))
    print(accuracy_score(y_test, predictions))

    # iterate over the confusion matrix elements and write all wrongly
    # predicted datarow into a csv file
    # TODO get back the original team names!
    misclassified_samples = X_test[y_test != predictions]
    output_csv = misclassified_samples.to_csv(index=False)
    f = open("prediction_outcome/wrongly_classified_testdata_" + str(datetime.datetime.now().time().hour) +
    str(datetime.datetime.now().time().minute) + str(datetime.datetime.now().time().second) + ".csv", "w")
    f.write(output_csv)
    f.write('This file was created on: ' + str(datetime.datetime.now()) + "\n")
    f.write(str(confusion_matrix(y_test, predictions)) + "\n")
    f.write('Accuracy Score: ' + str(accuracy_score(y_test, predictions)) + "\n")
    f.write(str(classification_report(y_test, predictions)))
    f.close()
    print(misclassified_samples)
    print(accuracy_score(y_test, predictions))


run_experiment()
