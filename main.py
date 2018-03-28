
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print('Le dataset')
dataset = pandas.read_csv('dataset/orders.csv')
dataset = dataset.drop(labels='eval_set', axis=1) #remove coluna eval_set

# remove linhas que possuem valor nam
dataset.dropna(inplace=True)

print('Separa X e Y')
array = dataset.values
X = array[:, 1:6] #cliente, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
y = array[:, 0]  #pedidos

# dividindo dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)

print('Spot-check')
# Algoritmos Spot-check
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    print(name)
    
    print('kfold')
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    print('cv_results')
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
    print('results.append')
    results.append(cv_results)
    print('name.append')
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


