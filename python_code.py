import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# from sklearn.SVM import SVC
from pandas.tools.plotting import scatter_matrix

dataset = pd.read_csv("Premier_League.csv")

############################# Data Cleaning ######################
#checking if any column contains null values
list_of_empty_columns = []
for c in dataset.columns:
    # print("Column name: ",c + " Clean?:",dataset[c].isnull().any())
    if dataset[c].isnull().any() == True:
        list_of_empty_columns.append(c)

if (len(list_of_empty_columns) > 0):
    print("List of list of empty columns",list_of_empty_columns)

############################ data exploration#####################
#total number of matches_drawn (Full time result)
total_matches = len(dataset["FTR"])

#number of matches won by home team
home_matches_won = len(dataset[dataset["FTR"]=="H"])

#number of matches won by away team
away_matches_won = len(dataset[dataset["FTR"]=="A"])

#number of matches drawn
matches_drawn = len(dataset[dataset["FTR"]=="D"])


#percentage of matches won by home team
percentage_home_matches_won = (home_matches_won/total_matches)*100
#percentage of matches won by away team
percentage_away_matches_won = (away_matches_won/total_matches)*100
#percentage of matches drawn
percentage_matches_drawn = (matches_drawn/total_matches)*100

print("Total number of matches :",total_matches)
print("home team % win :",percentage_home_matches_won)
print("away team % win:",percentage_away_matches_won)
print("draw %: ",percentage_matches_drawn)


# number of columns/features/predictor_variable
counter = 0
counter_using_list = []
for i in dataset.columns:
    counter_using_list.append(i)
    counter+=1
print("Number of features",len(counter_using_list))
print("Number of features",counter)

#getting the list of features with categorical and continous values
list_of_categorical_variables = []
list_of_continous_variables = []
for c in dataset.columns:
    if ((dataset[c].dtype)== object):
        list_of_categorical_variables.append(c)
    else:
        list_of_continous_variables.append(c)
print("list_of_categorical_variables = ",list_of_categorical_variables)
print("list_of_continous_variables = ",list_of_continous_variables)

#removing unwanted column like "Div, Date, HomeTeam, Away Team, HTR and Refree"
dataset = dataset.drop(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'HTR', 'Referee'], axis=1)
###################################Defining predictor(X) and target variable(y)#################
# featured_dataset =  dataset[list_of_continous_variables]

X = dataset.drop(["FTR"],axis=1)
y = dataset["FTR"]


##################################preprocessing the data#################
X = pd.get_dummies(X)

###################Splitting the data into train and test###############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

################################### Feature Scaling ###################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


###########################applying LogisticRegression model############
LogisticReg_classifier = LogisticRegression(random_state=0)
LogisticReg_classifier.fit(X_train, y_train)

y_pred = LogisticReg_classifier.predict(X_test)

#################################checking the accuracies#######
cm = confusion_matrix(y_test,y_pred)
score = LogisticReg_classifier.score(X_test, y_test)

# print(type(cm))
sum_of_diagonal_elements = sum(np.diagonal(cm))
sum_of_all_elements_confusion_matrix = np.sum(cm)
Accuracy_score = sum_of_diagonal_elements/sum_of_all_elements_confusion_matrix
print(Accuracy_score*100)
print(score*100)
# true_negative = cm[0,0]
# false_positive = cm[0,1]
# true_positive = cm[1,1]
# false_negative = cm[1,0]
# Accuracy_score = (true_negative+true_positive)/(np.sum(cm))*100
# print(Accuracy_score)

# True Positive, False Positive and similar counts and rates only make
# sense if there is a notion of "positive" and "negative" classes in
# your data. That is, only if you have exactly two classes.
# In our case we have three classes, not two.

# In our case, we can more or less reasonably discuss analogues,
# like "True H" numbers: take the number of cases we have correctly predicted
# classify as male and divide by the total number of males in the test
# sample.


#################checking cross_validation to avoid bias variance tradeoff##########
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

print(accuracies.mean()*100)




##############DecisionTreeClassifier##########
from sklearn.tree import DecisionTreeClassifier
DecisionTree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DecisionTree_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = DecisionTree_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)
# print(type(cm))
sum_of_diagonal_elements = sum(np.diagonal(cm))
sum_of_all_elements_confusion_matrix = np.sum(cm)
Accuracy_score = sum_of_diagonal_elements/sum_of_all_elements_confusion_matrix
print(Accuracy_score*100)


accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

print(accuracies.mean()*100)
