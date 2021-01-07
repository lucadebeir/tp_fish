import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

with open('./data/Fish.csv', newline='') as csvfile:
    dataset = pd.read_csv(csvfile, delimiter=';')
    print(dataset)

    print(dataset.keys())
    # [111 rows x 4 columns]
    # Index(['Species', 'Weight', 'Height', 'Width'], dtype='object')

    # print dimensionality of the DataFrame
    print(dataset.shape)
    # (111, 4)

    # print the first n lines of the object according to position
    print(dataset.head())
    #   Species  Weight   Height   Width
    # 0        0   242.0  11.5200  4.0200
    # 1        0   290.0  12.4800  4.3056
    # 2        0   340.0  12.3778  4.6961
    # 3        0   363.0  12.7300  4.4555
    # 4        0   430.0  12.4440  5.1340

    # print a concise summary of a DataFrame
    print(dataset.info())

    # print important statistical indicators
    print(dataset.describe())

    # each numerical characteristic is plotted against each other
    sns.pairplot(dataset, hue='Species', palette='viridis')

    # print a visualization of the data
    dataset.hist(bins=50, figsize=(20, 15))
    plt.show()

    # Calculate the median value for Weight
    median_weight = dataset['Weight'].median()
    # Substitute it in the Weight column of the
    # dataset where values are 0
    dataset['Weight'] = dataset['Weight'].replace(to_replace=0, value=median_weight)

    # Visualising the data
    # dataset.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # FIRST APPROACHES
    print('---- FIRST APPROACHES ----')

    # print correlation matrix
    corr = dataset.corr()
    print(corr)

    # print a different visualization of the correlation matrix
    sns.heatmap(corr, annot=True)
    plt.show()

    # splitting dataset : split the training dataset in 80% / 20%
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    # print(train_set)
    # print(test_set)
    # separate labels from the rest of the dataset
    train_set_labels = train_set["Species"].copy()
    train_set = train_set.drop("Species", axis=1)

    test_set_labels = test_set["Species"].copy()
    test_set = test_set.drop("Species", axis=1)

    # apply a scaler
    scaler = Scaler()
    scaler.fit(train_set)
    train_set_scaled = scaler.transform(train_set)
    test_set_scaled = scaler.transform(test_set)
    df = pd.DataFrame(data=train_set_scaled)

    # model params
    param_grid = {
        'C': [1.0, 10.0, 50.0],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'shrinking': [True, False],
        'gamma': ['auto', 1, 0.1],
        'coef0': [0.0, 0.1, 0.5]
    }

    # apply svc algo with params
    model_svc = SVC()
    grid_search = GridSearchCV(
        model_svc, param_grid, refit=True, verbose=3, cv=10, scoring='accuracy')
    grid_search.fit(train_set_scaled, train_set_labels)
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)
    # print the best score found
    print(grid_search.best_score_)  # 0.8638888888888889

    # apply the parameters to the model and train it
    # create an instance of the algorithm using parameters from best_estimator_ property
    svc = grid_search.best_estimator_
    # use the train dataset to train the model
    X = train_set_scaled
    Y = train_set_labels

    # train the model for the prediction
    svc.fit(X, Y)
    print(svc)

    # PREDICTION
    print('---- PREDICTION ----')
    # We create a new (fake) fish
    new_df = pd.DataFrame([[450, 11, 5]])  # LINE TO CHANGE
    # We scale those values like the others
    new_df_scaled = scaler.transform(new_df)
    # We predict the outcome
    prediction = svc.predict(new_df_scaled)
    # A value of "1" means that this fish is probably the one studied
    print(prediction)

    # SECOND APPROACHES
    print('---- SECOND APPROACHES ----')

    # separate labels from the rest of the dataset
    X = dataset.drop('Species', axis=1)
    y = dataset['Species']
    # print(X)
    # print(y)

    # splitting dataset : split the training dataset in 80% / 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # apply a scaler
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # import the model and instance it, then train the model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # print the classification report and the confusion matrix
    print(classification_report(y_test, lr.predict(X_test)))
    print(confusion_matrix(y_test, lr.predict(X_test)))
