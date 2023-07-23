import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,pointbiserialr
from scipy.stats import ttest_rel
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from scipy.special import softmax
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(6.2,3)




seeds=np.arange(1,11)


df_res=[]
for dist in ['normal','pareto']:
	dataset_files=['../output_replicated/pred_rel/matched_test_{}_{}.csv']
	corr_arr=[]
	corr_arr_f=[]
	corr_arr_m=[]
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(6.2,3)

	df_seed=[]

	for seed in seeds:
		df=pd.read_csv(dataset_files[0].format(dist,
			seed))
		df_all = pd.read_csv('../../example/synthetic/synthetic_test_{}.csv'.format(
			dist))
		all_val=df.values[0].squeeze()
		assert df.values[0].squeeze().shape[0]==df_all.shape[0]
		df_all=df_all.iloc[0:len(all_val)]
		df_all['prediction']=StandardScaler().fit_transform(softmax(all_val[0:df_all.shape[0]]).reshape(-1, 1))
		df_all['id']=np.arange(df_all.shape[0])
		df_seed.append(df_all)
	df_seed=pd.concat(df_seed).reset_index()
	df_seed_f=df_seed[df_seed.G=='F']
	df_seed_m=df_seed[df_seed.G=='M']

	df_seed=df_seed.groupby('id').std().reset_index()
	mean_std_all=df_seed['prediction'].mean()

	df_seed_f=df_seed_f.groupby('id').std().reset_index()
	mean_std_f=df_seed_f['prediction'].mean()


	df_seed_m=df_seed_m.groupby('id').std().reset_index()
	mean_std_m=df_seed_m['prediction'].mean()

	df_res.append(pd.DataFrame({'dist':dist,
		'all':[mean_std_all],
		'f':mean_std_f,
		'm':mean_std_m}))




for dist in ['fairtrecnotpretrained']:
	dataset_files=['../output_replicated/pred_rel/matched_test_{}_{}.csv']
	corr_arr=[]
	corr_arr_f=[]
	corr_arr_m=[]
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(6.2,3)

	df_seed=[]

	for seed in seeds:
		corr_arr_seed=[]
		corr_arr_f_seed=[]
		corr_arr_m_seed=[]
		df=pd.read_csv(dataset_files[0].format(dist,
			seed))
		df_all = pd.read_csv('../../example/fairtrecnotpretrained/fairtrecnotpretrained_test.csv')
		assert df.values[0].squeeze().shape[0]==df_all.shape[0]
		all_val=df.values[0].squeeze()
		df_all=df_all.iloc[0:len(all_val)]
		df_all['prediction']=StandardScaler().fit_transform(softmax(all_val[0:df_all.shape[0]]).reshape(-1, 1))
		df_all['id']=np.arange(df_all.shape[0])
		df_seed.append(df_all)
	# plt.title('{}: Inferred vs True Relevance'.format(dist),fontsize=22)
	df_seed=pd.concat(df_seed).reset_index()
	df_seed=df_seed.groupby('id').std().reset_index()
	mean_std_all=df_seed['prediction'].mean()

	df_res.append(pd.DataFrame({'dist':dist,
		'all':[mean_std_all],
		'f':0,
		'm':0}))


df_res=pd.concat(df_res)
df_res.to_csv('stability.csv',index=False)







