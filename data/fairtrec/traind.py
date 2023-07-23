import pandas as pd


from sklearn.svm import SVC

df=pd.read_csv('fairtrecnotpretrained_train_all.csv')
df_test=pd.read_csv('fairtrecnotpretrained_test.csv')
cols=['0','1','2']
clf=SVC()
X=df[cols].values
y=df['relevance'].values
X_test=df_test[cols].values
y_test=df_test['relevance'].values
clf.fit(X,y)
print(clf.score(X_test, y_test))
