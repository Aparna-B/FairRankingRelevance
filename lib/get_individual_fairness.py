import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,wilcoxon,kruskal
from scipy.special import softmax
import math

def get_individual_unfairness_geometric_attention(df_name, cat, p=0.68,k=10,
	use_geom="Log",
	worthiness='pred'):
	df_all = []
	for seed in seeds:
		df = pd.read_csv(df_name.format(cat,seed))

		if 'normal' in df_name:
			test_file = '../../example/synthetic/synthetic_test_normal.csv'
			score_col = 'U'
			sensitive_col = 'G'
		elif 'pareto' in df_name:
			test_file = '../../example/synthetic/synthetic_test_pareto.csv'
			score_col = 'U'
			sensitive_col = 'G'

		elif 'fairtrecnotpretrained' in df_name:
			test_file = '../../example/fairtrecnotpretrained/fairtrecnotpretrained_test.csv'
			score_col = 'relevance'
			sensitive_col = 'country'
			ranked_df = pd.read_csv(test_file)

		elif 'fairtrec4' in df_name:
			test_file = '../../example/fairtrec4/fairtrec4_test.csv'
			score_col = 'relevance'
			sensitive_col = 'country'
			ranked_df = pd.read_csv(test_file)

		else:
			test_file = '../../example/jee/jee_test.csv'
			score_col = 'total_score'
			sensitive_col = 'gender'
		ranked_df = pd.read_csv(test_file)
		if 'total_score' in ranked_df.columns:
			ranked_df.rename(columns={'total_score':'U'},inplace=True)


		if 'relevance' in ranked_df.columns:
			ranked_df.rename(columns={'relevance':'U'},inplace=True)


		# NB: to standardize that minority/majority in fairness calc
		if 'fairtrec' in df_name:
			ranked_df[sensitive_col]=1-ranked_df[sensitive_col].values

		ranked_df['idx'] = np.arange(ranked_df.shape[0])
		ranked_df['pred']=df.values[0,:]
		ranked_df['pred']=softmax(ranked_df.pred.values.reshape(-1, 1))
		#ranked_df['pred']=ranked_df['pred'].values/ranked_df['pred'].sum()
		ranked_df['U']=MinMaxScaler().fit_transform(ranked_df.U.values.reshape(-1, 1))
		ranked_df['U']=ranked_df['U'].values/ranked_df['U'].sum()
		if use_geom=='Geom':
			loc_weights=[1 for i in range(k+1)[1:]] #p*pow(1-p, i-1)
		elif use_geom=='Log':
			loc_weights=[1/math.log2(i+1) for i in range(k+1)[1:]]
		else:
			print('here')
			loc_weights=[1 for i in range(k+1)[1:]]
		for i in range(k, ranked_df.shape[0]):
			loc_weights.append(0)
		loc_weights = np.array(loc_weights)
		loc_weights = np.array(loc_weights)/np.sum(loc_weights)
		ranked_df=ranked_df.sort_values('pred',ascending=False)
		ranked_df['loc_weights']=loc_weights
		df_all.append(ranked_df)
	df_all = pd.concat(df_all)
	df_all = df_all.groupby('idx').sum().reset_index()
	df_all['diff']=np.abs(df_all['loc_weights'].values-df_all[worthiness].values)
	df_all=df_all.sort_values('idx')
	return df_all['diff'].sum(),df_all['diff'].std(), df_all['diff'].values



# defining the parameters for output
p=0.68
k=10
fairness_metric = 'exposure_fairness'
worthiness='U'

seeds=np.arange(1,11,1)

## 1.
print('Varying click model')
dataset_files=[
'../output_replicated/pred_rel/{}_test_normal_{}.csv',
'../output_replicated/pred_rel/{}_test_pareto_{}.csv',
'../output_replicated/pred_rel/{}_test_fairtrecnotpretrained_{}.csv']

fairness_click_dict={}
fairness_click_dict_sd={}
tmp_dict={}
tmp_dict['matched']=0
for dataset in dataset_files:
	fairness_click_dict[dataset]={}
	fairness_click_dict_sd[dataset]={}
count=0
tru_score = []

for dataset in dataset_files:
	for cat in ['matched']:
		curr_score=[]
		for worthiness in ['pred','U']:
			fairness_score,fairness_score_sd,fairness_per_item=get_individual_unfairness_geometric_attention(dataset,
				cat, 
					p=p,k=k,
					worthiness=worthiness
					)
			fairness_click_dict[dataset][worthiness]=fairness_per_item
			fairness_click_dict_sd[dataset][worthiness]=fairness_score_sd
			if cat=='matched':
				tru_score.append(fairness_score)
			else:
				count=1
				# print(ttest_rel(tru_score[count], fairness_score))
				# print(tru_score[count], fairness_score)
print(fairness_click_dict)
print(fairness_click_dict_sd)
for dataset in dataset_files:
	print(dataset,wilcoxon(fairness_click_dict[dataset]['U'],
		fairness_click_dict[dataset]['pred']))

for dataset in dataset_files:
	print(dataset,np.sum(fairness_click_dict[dataset]['U']),
		np.sum(fairness_click_dict[dataset]['pred']))





