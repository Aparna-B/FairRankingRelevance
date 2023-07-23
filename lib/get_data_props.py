import pandas as pd
import numpy as np

df_train=pd.read_csv('../example/fairtrecnotpretrained/data/train_fairtrecnotpretrained/train_fairtrecnotpretrained.txt',header=None,sep=" ")
df_val=pd.read_csv('../example/fairtrecnotpretrained/data/val_fairtrecnotpretrained/val_fairtrecnotpretrained.txt',header=None,sep=" ")
df_test=pd.read_csv('../example/fairtrecnotpretrained/data/test_fairtrecnotpretrained/test_fairtrecnotpretrained.txt',header=None,sep=" ")

count_dict={}
df_sel=pd.concat([df_train,df_val,df_test])
for i in np.sort(df_sel[0].unique()):
	counts=df_sel[df_sel[0]==i].shape[0]/df_sel.shape[0]
	count_dict[i]=counts

print(count_dict)



for dist in ['normal','pareto']:
	df_train=pd.read_csv('../example/synthetic/data/train_{}/train_{}.txt'.format(dist,dist),header=None,sep=" ")
	df_val=pd.read_csv('../example/synthetic/data/val_{}/val_{}.txt'.format(dist,dist),header=None,sep=" ")
	df_test=pd.read_csv('../example/synthetic/data/test_{}/test_{}.txt'.format(dist,dist),header=None,sep=" ")

	count_dict={}
	df_sel=pd.concat([df_train,df_val,df_test])
	for i in np.sort(df_sel[0].unique()):
		counts=df_sel[df_sel[0]==i].shape[0]/df_sel.shape[0]
		count_dict[i]=counts

	print(count_dict)

