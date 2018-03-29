import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

##### Leitura dataset #####
orders = pd.read_csv('dataset/orders.csv')
products = pd.read_csv('dataset/products.csv')
# remove linhas que possuem valor nam
products.dropna(inplace=True)
# remove linhas que possuem valor nam
orders.dropna(inplace=True)

orders_train = orders.iloc[orders.values[:, 2] == 'train', :]
orders_prior = orders.iloc[orders.values[:, 2] == 'prior', :]

orders_train = orders_train.drop('eval_set', 1)
orders_prior = orders_prior.drop('eval_set', 1)

##### Treinamento #####
order_products = pd.read_csv('dataset/order_products__train.csv')
# remove linhas que possuem valor nam
order_products.dropna(inplace=True)

dataset = pd.merge(order_products, orders_train, on='order_id', how='outer')
dataset = pd.merge(dataset, products, on='product_id', how='outer')

dataset = dataset.drop(['add_to_cart_order', 'order_number', 'reordered', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'aisle_id', 'department_id'], axis=1) 

#print(dataset.head(20))

score =  dataset.groupby(['product_id']).size().reset_index(name='counts')