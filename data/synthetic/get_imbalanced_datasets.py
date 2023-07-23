import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler,KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os 
import random
SEED=1
np.random.seed(SEED)
random.seed(SEED)

def get_all_txt_rows(df, col, filename,upper_quartile):
	df_curr=df.copy()
	text_str=[]
	text_form_str="{} qid:1 1:{} 2:{}"

	

	total=df.shape[0]
	for idx, row in df_curr.iterrows():
		curr_rel=int(row[col])
		curr_txt=text_form_str.format(curr_rel,
			row['X'],
			row['Y'])
		text_str.append(curr_txt)
	text_str = np.array(text_str)
	# random.shuffle(text_str)

	with open(filename, 'w') as f:
	    for item in text_str:
	        f.write("%s\n" % item)


if __name__ == '__main__':
	os.makedirs('data',exist_ok=True)
	for dist in ['normal','pareto']:

		os.makedirs('data/train_{}'.format(dist),exist_ok=True)
		os.makedirs('data/val_{}'.format(dist),exist_ok=True)
		os.makedirs('data/test_{}'.format(dist),exist_ok=True)


		df=pd.read_csv('R{}.csv'.format(dist))
		min_val=np.min(df['U'].values)
		print(df['U'].min(),df['U'].mean(),df['U'].std(),df['U'].max())

		if min_val<0:
			assert dist=='normal'
			df['U'] = df['U'].values+2

		if dist=='pareto':
			df['U']=np.log2(df['U'].values)


		df_f = df[df.G=='F']
		df_m = df[df.G=='M']

		for imb_ratio in np.arange(5,10)/10:
			df = pd.concat(
				[
				df_f.sample(int((1-imb_ratio)*25000), replace=False),
				df_m.sample(int(imb_ratio*25000), replace=False)
				]
				)
			df = df.sample(frac=1, random_state=1).reset_index(drop=True)
			train_idx, test_idx=train_test_split(np.arange(df.shape[0]), test_size=0.2)
			train_idx, val_idx=train_test_split(train_idx,test_size=1/8)


			df_train=df.iloc[train_idx]
			df_val=df.iloc[val_idx]
			df_test=df.iloc[test_idx]
			upper_quartile=np.percentile(df_train['U'].values, 50)
			

			print(upper_quartile)
			for col in ['X','Y']:
				scaler=RobustScaler()
				scaler = scaler.fit(df_train[col].values.reshape(-1, 1))
				df_train[col] = scaler.transform(df_train[col].values.reshape(-1, 1))
				df_val[col] = scaler.transform(df_val[col].values.reshape(-1, 1))
				df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))

			
			df_train_pre_bin=df_train.copy()
			for col in ['U']:
				scaler=KBinsDiscretizer(n_bins=5,encode='ordinal', strategy='uniform')
				scaler = scaler.fit(df_train[col].values.reshape(-1, 1))
				df_train[col] = scaler.transform(df_train[col].values.reshape(-1, 1))
				df_val[col] = scaler.transform(df_val[col].values.reshape(-1, 1))
				df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))
			print(np.unique(df_train[col].values))
			df_train.to_csv('synthetic{}_{}_train.csv'.format(imb_ratio, dist),index=False)
			# df_test['U']=np.array(df_test['U'].values>upper_quartile,dtype=np.int8)
			df_test.to_csv('synthetic{}_test_{}.csv'.format(imb_ratio, dist),index=False)
			df_sel=pd.concat([df_train,df_val,df_test])
			print(df_test.shape, df_val.shape)
			count_dict={}
			for i in df_sel.U.unique():
				counts=df_sel[df_sel.U==i].shape[0]/df_sel.shape[0]
				count_dict[i]=counts

			sns.distplot(df_train[df_train.G=='M']['U'].values)
			sns.distplot(df_train[df_train.G=='F']['U'].values)

			plt.legend(['Male','Female'])
			plt.title('Relevance distribution (normalized): {}'.format(dist))
			plt.savefig('{}_rel_hist.png'.format(dist),dpi=300)

			get_all_txt_rows(df_train,'U','data/train{}_{}/train{}_{}.txt'.format(imb_ratio, dist,imb_ratio,dist),upper_quartile)
			get_all_txt_rows(df_val,'U','data/val{}_{}/val{}_{}.txt'.format(imb_ratio,dist,imb_ratio,dist),upper_quartile)
			get_all_txt_rows(df_test,'U','data/test{}_{}/test{}_{}.txt'.format(imb_ratio,dist,imb_ratio,dist),upper_quartile)











