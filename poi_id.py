#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.

features_list = ['poi', 'salary','to_messages', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person',
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#People in the dataset
people = len(data_dict)
print "There are " + str(people) + " people in the dataset."

#Features in the dataset
features = len(data_dict['SKILLING JEFFREY K'])
print "There are " + str(features) + " features in the dataset."

#poi's in the dataset
def poi_counter(file):
    count = 0
    for person in file:
        if file[person]['poi'] == True:
            count += 1
    print "There are " + str(count) + " poi's in the dataset."
    print "There are " + str(people - count) + " non-poi's in the dataset."

poi_counter(data_dict)
print

#this function generates a csv file of our dataset so you can explore it freely
def generate_csv():
    import csv

    data_dict.pop('TOTAL', None)
    with open('Empdata.csv', 'w') as csvfile:
        fieldnames = ['name']
        fieldnames = fieldnames + data_dict.itervalues().next().keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for name in data_dict:
            details = data_dict[name]
            details['name'] = name
            writer.writerow(details)

# generate_csv()

### Task 2: Remove outliers
import matplotlib.pyplot

features = ["poi","salary", "bonus"]
my_dataset = data_dict
my_dataset.pop('TOTAL', 0)
data_temp = featureFormat(my_dataset, features)

cleaned_data = []
cleaned_data = data_temp
cleaned_data = sorted(cleaned_data, key=lambda tup: tup[1])

#Removing the top 5% of the people with highest salaries
totalV = len(cleaned_data) * 0.95
cleaned_data_temp = cleaned_data[:int(totalV)]

for i in range(int(totalV), len(cleaned_data) ): #if there is a poi in that 5%, add it back
    if cleaned_data[i][0]==1:
        cleaned_data_temp.append(cleaned_data[i])

cleaned_data = sorted(cleaned_data_temp, key=lambda tup: tup[2])

#Removing the top 5% of the people with highest bonus
totalV = len(cleaned_data) * 0.95
cleaned_data_temp = cleaned_data[:int(totalV)]

for i in range(int(totalV), len(cleaned_data) ): #if there is a poi in that 5%, add it back
    if cleaned_data[i][0]==1:
        cleaned_data_temp.append(cleaned_data[i])

#Before taking out outliers
for point in data_temp:
    salary = point[1]
    bonus = point[2]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()

#After taking out outliers
for point in cleaned_data_temp:
    salary = point[1]
    bonus = point[2]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()

from collections import defaultdict

missingEmp = defaultdict(int)


for name in data_dict:
    for feature in features_list:
        if data_dict[name][feature] == 'NaN':
            missingEmp[name] += 1


emptyEntries = []
for name in data_dict:
    if missingEmp[name] >= 17 and data_dict[name]['poi']==False:
        emptyEntries.append(name)

for name in emptyEntries:
    data_dict.pop(name, None)

print "People who are not POI and have 17 or more missing Features: "
print emptyEntries
print


### Task 3: Create new feature(s)

for name in my_dataset:

    if my_dataset[name]['to_messages'] != 'NaN' and my_dataset[name]['from_messages'] != 'NaN':
        total_messages = my_dataset[name]['to_messages'] + my_dataset[name]['from_messages']

    if my_dataset[name]['from_poi_to_this_person'] != 'NaN' and my_dataset[name]['from_this_person_to_poi'] != 'NaN':
        poi_messages = my_dataset[name]['from_poi_to_this_person'] + my_dataset[name]['from_this_person_to_poi']

    if my_dataset[name]['to_messages'] != 'NaN' and my_dataset[name]['from_messages'] != 'NaN' and \
        my_dataset[name]['from_poi_to_this_person'] != 'NaN' and my_dataset[name]['from_this_person_to_poi'] != 'NaN':
        my_dataset[name]['poi_ratio'] = round((float(poi_messages) / float(total_messages)), 5)
        #print my_dataset[name]['poi_ratio']

    else:
        my_dataset[name]['poi_ratio'] = 0


    if my_dataset[name]['to_messages'] != 'NaN' and my_dataset[name]['from_poi_to_this_person'] != 'NaN':
        my_dataset[name]['poi_to_ratio'] = round(float(my_dataset[name]['from_poi_to_this_person']) / \
                                           float(my_dataset[name]['to_messages']), 5)
    else:
        my_dataset[name]['poi_to_ratio'] = 0

        # print my_dataset[name]['poi_to_ratio']

    if my_dataset[name]['from_messages'] != 'NaN' and my_dataset[name]['from_this_person_to_poi'] != 'NaN':
        my_dataset[name]['poi_from_ratio'] = round(float(my_dataset[name]['from_this_person_to_poi']) / \
                                           float(my_dataset[name]['from_messages']), 5)
        # print my_dataset[name]['poi_from_ratio']

    else:
        my_dataset[name]['poi_from_ratio'] = 0


modified_features_list = features_list + ['poi_from_ratio']
modified_features_list = modified_features_list + ['poi_to_ratio']
modified_features_list = modified_features_list + ['poi_ratio']

#comment out this line if you want to run the algorithm on original features
features_list = modified_features_list

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

#Feature Scaling
#The method MinMaxScaler from the sci-kit learn library will be used to normalise our features

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

#Feature Selection
from sklearn.feature_selection import SelectKBest

#The method SelectKbest from the sci-kit learn library will be used to select and determine the best features

num_features = 10
k_best = SelectKBest(k=num_features)
k_best.fit(features, labels)

scores = zip(features_list[1:], k_best.scores_)

from operator import itemgetter
best_features = sorted(scores, key=itemgetter(1), reverse=True)
print "Best Features and their respective scores sorted in Ascending order: "
print best_features

count = -1
best =[]
for k, v in best_features:
    count+=1
    if count < num_features:
        best.append(k)

features_list = ['poi'] + best


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# from sklearn.svm import SVC
# clf = SVC()

from sklearn import tree

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# clf = AdaBoostClassifier(DecisionTreeClassifier())

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(random_state=1, n_estimators=50, criterion="entropy", min_samples_split=50)


clf = tree.DecisionTreeClassifier(min_samples_split = 2, random_state= 2, criterion= 'entropy', max_depth= 4,
                                  min_samples_leaf= 4)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
# Because of the small size of the dataset, the script uses stratified shuffle split cross validation.

def tune_algo():
    from sklearn import grid_search

    parameters = {'criterion': ('gini', 'entropy'), 'max_depth': [1, 2, 3, 4, 5], 'min_samples_split':
        [2, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50], 'min_samples_leaf': [1, 2, 3, 4, 5], 'random_state': [1, 2, 3, 4, 5]}

    tre = tree.DecisionTreeClassifier()
    clfgrid = grid_search.GridSearchCV(tre, parameters)
    clfgrid.fit(features, labels)
    print "Best Combination of Parameters: "
    print clfgrid.best_params_

# tune_algo()

#For Cross validation, i'm
from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(labels, 1000, test_size=0.3, random_state=42)

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in sss:
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        labels_test.append(labels[jj])

    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1

total_predictions = true_negatives + false_negatives + false_positives + true_positives
accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
precision = 1.0 * true_positives / (true_positives + false_positives)
recall = 1.0 * true_positives / (true_positives + false_negatives)

print "Accuracy: ", accuracy
print "Precision: ", precision
print "Recall: ", recall

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results.

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

dump_classifier_and_data(clf, my_dataset, features_list)