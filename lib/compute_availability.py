import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,pointbiserialr,ks_2samp
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_rel
import matplotlib
import seaborn as sns
from scipy.special import softmax
fig = matplotlib.pyplot.gcf()

seeds=np.arange(1,11)
FS=20

for dist in ['normal','pareto','fairtrecnotpretrained']:
	dataset_files=['../output_replicated/pred_rel/matched_test_{}_{}.csv']
	corr_arr=[]
	fig = matplotlib.pyplot.gcf()

	df_seed=[]
	for seed in seeds:
		df=pd.read_csv(dataset_files[0].format(dist,
			seed))
		if 'fairtrec' in dist:
			df_all = pd.read_csv('../../example/fairtrecnotpretrained/fairtrecnotpretrained_test.csv'.format(
				dist))
			score_col='relevance'

		else:
			df_all = pd.read_csv('../../example/synthetic/synthetic_test_{}.csv'.format(
				dist))
			score_col='U'
		df_all['ind']=np.arange(df_all.shape[0])
		all_val=df.values[0].squeeze()
		df_all=df_all.iloc[0:len(all_val)]
		df_all['prediction']=MinMaxScaler().fit_transform(softmax(all_val[0:df_all.shape[0]]).reshape(-1,1))
		df_all['seed']=seed
		df_seed.append(df_all)
	df_seed=pd.concat(df_seed).reset_index()
	df_seed=df_seed.groupby('ind').mean().reset_index()
	plt.hist(df_seed['prediction'].values,bins=20)
	plt.hist(MinMaxScaler().fit_transform(df_seed[score_col].values.reshape(-1,1)),bins=20)
	plt.legend(['Pred','True'],fontsize=FS)
	plt.xlabel('Relevance',fontsize=FS)
	plt.ylabel('#Test Set Items',fontsize=FS)
	plt.xticks(fontsize=FS)
	plt.yticks(fontsize=FS)

	plt.tight_layout()
	plt.savefig('corr_{}_hist_plot.png'.format(dist),dpi=300,bbox_inches="tight")
	plt.close()
	a=df_seed['prediction'].values.tolist()
	b=MinMaxScaler().fit_transform(df_seed[score_col].values.reshape(-1,1))[:,0].tolist()
	print(dist, ks_2samp(a,b))


