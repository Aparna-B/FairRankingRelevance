import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os 
import random
SEED=1
np.random.seed(SEED)
random.seed(SEED)

try:
	os.makedirs('data/train_fairtrecnotpretrained',exist_ok=True)
	os.makedirs('data/val_fairtrecnotpretrained',exist_ok=True)
	os.makedirs('data/test_fairtrecnotpretrained',exist_ok=True)
except FileExistsError:
	pass
	print('NOTE: All output folders exist!')


def get_which_quartile(row, df, col):
	lower_25 = np.quantile(df[col].values, 0.25)
	mid = np.quantile(df[col].values, 0.50)
	upper_75 = np.quantile(df[col].values, 0.75)
	quar_arr = np.array([lower_25,mid,upper_75])

	return np.argmin(np.abs(quar_arr-row[col]))

	        
def get_all_txt_rows(df, col, filename):
	df_curr=df.copy()
	
	text_str=[]
	text_form_str="{} qid:1 1:{} 2:{} 3:{}"

	total=df.shape[0]
	for idx, row in df_curr.iterrows():
		curr_rel=int(row[col])
		curr_txt=text_form_str.format(curr_rel,
			row['0'],
			row['1'],
			row['2'])
		text_str.append(curr_txt)
	text_str = np.array(text_str)
	# random.shuffle(text_str)

	with open(filename, 'w') as f:
	    for item in text_str:
	        f.write("%s\n" % item)
	        


if __name__ == '__main__':
	df=pd.read_csv('fairtrec_query41_v2.csv')
	df=df[df.country.isin([0,1])]

	plt.hist(df[df.country==0]['relevance'].values)
	plt.hist(df[df.country==1]['relevance'].values)

	plt.legend(['Europe','Not Europe'])
	plt.title('Relevance distribution (unnormalized): {}'.format('Fairtrec'))
	plt.ylabel('#Documents')
	plt.xlabel('Relevance Score (total score in the JEE exam)')
	plt.savefig('{}_rel_hist.png'.format('fairtrec'),dpi=300)



	print(df.iloc[0])
	#df = df.sample(frac=1, random_state=1).reset_index(drop=True)
	np.random.seed(1)

	train_idx, test_idx=train_test_split(df.doc_id, test_size=0.2)
	train_idx, val_idx=train_test_split(train_idx,test_size=1/8)

	embeddings=np.load('pred_arr_query_6.npy', allow_pickle=True)
	embeddings=np.vstack(embeddings)
	df_embed=pd.DataFrame(embeddings)
	embed_cols=df_embed.columns
	print(len(embed_cols))
	assert len(embed_cols)==128

	doc_arr=np.load('doc_id_arr_query_6.npy')
	df_embed['doc_id']=doc_arr

	df_train=df[df.doc_id.isin(train_idx)].sort_values('doc_id')
	df_val=df[df.doc_id.isin(val_idx)].sort_values('doc_id')
	df_test=df[df.doc_id.isin(test_idx)].sort_values('doc_id')

	train_embed=df_embed[df_embed.doc_id.isin(train_idx)].sort_values('doc_id')
	val_embed=df_embed[df_embed.doc_id.isin(val_idx)].sort_values('doc_id')
	test_embed=df_embed[df_embed.doc_id.isin(test_idx)].sort_values('doc_id')


	assert (df_train.doc_id.values==train_embed.doc_id.values).all()
	assert (df_val.doc_id==val_embed.doc_id).all()
	assert (df_test.doc_id==test_embed.doc_id).all()


	pca = PCA(n_components=3)
	train_embed=pca.fit_transform(train_embed[embed_cols].values)
	val_embed=pca.transform(val_embed[embed_cols].values)
	test_embed=pca.transform(test_embed[embed_cols].values)

	new_cols=['0','1','2']
	df_train[new_cols]=train_embed
	df_val[new_cols]=val_embed
	df_test[new_cols]=test_embed

	df_train = df_train.sample(frac=1, random_state=1).reset_index(drop=True)
	df_val = df_val.sample(frac=1, random_state=1).reset_index(drop=True)
	df_test = df_test.sample(frac=1, random_state=1).reset_index(drop=True)


	for col in ['0','1','2']:
		scaler = RobustScaler()
		scaler = scaler.fit(df_train[col].values.reshape(-1, 1))
		df_train[col] = scaler.transform(df_train[col].values.reshape(-1, 1))
		df_val[col] = scaler.transform(df_val[col].values.reshape(-1, 1))
		df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1))



	df_test.to_csv('fairtrecnotpretrained_test.csv',index=False)
	df_train.to_csv('fairtrecnotpretrained_train_all.csv',index=False)

	get_all_txt_rows(df_train,'relevance','data/train_fairtrecnotpretrained/train_fairtrecnotpretrained.txt')
	get_all_txt_rows(df_val,'relevance','data/val_fairtrecnotpretrained/val_fairtrecnotpretrained.txt')
	get_all_txt_rows(df_test,'relevance','data/test_fairtrecnotpretrained/test_fairtrecnotpretrained.txt')








