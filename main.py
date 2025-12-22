#loading up the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#creating dataframe
df=pd.read_csv('spam.csv', encoding='latin-1')

#cleaning df
df=df[['v1','v2']]
df.columns=['label','message']

#looking into the data
print(df['label'].value_counts())

df['length']=df['message'].apply(len)
plt.figure(figsize=(10,6))
sns.histplot(df,x='length',hue='label',bins=50)
plt.show()

#data splitting
x=df['message']
y=df['label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

#setting up tfidf
tfidf=TfidfVectorizer(stop_words='english')
X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)

#using gridsearch for optimizing the model
grid=GridSearchCV(LinearSVC(),param_grid={'C':[0.1,1,10,100,1000]},verbose=3)
grid.fit(X_train_tfidf,y_train)
print(grid.best_params_)

#model fitting/how it can be improved
svmodel=LinearSVC(C=10)
svmodel.fit(X_train_tfidf,y_train)

#saving the model and tfidf
with open('model.pkl','wb') as file:
    pickle.dump(svmodel,file)

with open('tfidf.pkl','wb') as file2:
    pickle.dump(tfidf,file2)

prediction=svmodel.predict(X_test_tfidf)

#performance measures
print(confusion_matrix(y_test,prediction))
print('\n')
print(classification_report(y_test,prediction))