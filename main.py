import numpy as np
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

##### Leitura dataset #####
orders = pandas.read_csv('dataset/orders.csv')
products = pandas.read_csv('dataset/products.csv')
# remove linhas que possuem valor nam
products.dropna(inplace=True)
# remove linhas que possuem valor nam
orders.dropna(inplace=True)

orders_train = orders.iloc[orders.values[:, 2] == 'train', :]
orders_prior = orders.iloc[orders.values[:, 2] == 'prior', :]

##### Treinamento #####
order_products = pandas.read_csv('dataset/order_products__train.csv')
# remove linhas que possuem valor nam
order_products.dropna(inplace=True)

dataset_order = pandas.merge(order_products, orders_train, on='order_id', how='outer')
dataset_products = pandas.merge(order_products, products, on='product_id', how='outer')

#print(products_orders.head(20))