import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,pointbiserialr,kruskal
from scipy.stats import ttest_rel
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import ndcg_score,dcg_score

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(6.2,3)
FS=20


seeds=np.arange(1,11)


for dist in ['normal','pareto','fairtrecnotpretrained']:
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
		if 'fairtrec' in dist:
			df_all = pd.read_csv('../../example/fairtrecnotpretrained/fairtrecnotpretrained_test.csv'.format(
			dist))
			score_col='relevance'
		else:
			df_all = pd.read_csv('../../example/synthetic/synthetic_test_{}.csv'.format(
			dist))
			score_col='U'
		print(df.shape)
		all_val=df.values[0].squeeze()
		# df_all=df_all.iloc[0:len(all_val)]
		assert df_all.shape[0]==len(all_val)
		df_all['prediction']=softmax(all_val[0:df_all.shape[0]])
		df_all['ind']=np.arange(df_all.shape[0])
		df_all['true_rank']=df_all[score_col].values
		df_all['seed']=seed
		df_seed.append(df_all)
	df_seed=pd.concat(df_seed).reset_index()
	# NB: Similar trends are observed on averaging across 10 runs for all items,
	# but differences are larger.
	# df_seed=df_seed.groupby('ind').mean().reset_index()

	sns.boxplot(x="true_rank",y="prediction",data=df_seed)
	plt.tight_layout()
	plt.xlabel('Relevance label',fontsize=FS)
	plt.ylabel('Predicted relevance',fontsize=FS)
	if 'fairtrec' in dist:
		plt.xticks(np.arange(2),np.arange(2),fontsize=FS)
	else:
		plt.xticks(np.arange(5),np.arange(5),fontsize=FS)
	plt.yticks(fontsize=16)
	plt.ylim([0.00004,0.00016])
	plt.tight_layout()
	plt.savefig('corr_{}_box.png'.format(dist),dpi=300,bbox_inches="tight")
	plt.close()

	# NB: Similar trends are observed on averaging across 10 runs for all items.
	df_seed=df_seed.groupby('ind').mean().reset_index()
	group_level_dists=[]
	for rel_grade in df_seed.true_rank.unique():
		df_curr=df_seed[df_seed.true_rank==rel_grade]
		group_level_dists.append(df_curr.prediction.values)
	if 'fairtrec' in dist:
		print('For dist: {}, kruskal p: {}'.format(dist, kruskal(group_level_dists[0],
		group_level_dists[1])[1]))
		assert 	kruskal(group_level_dists[0],group_level_dists[1])[1]<=0.05
	else:
		print('For dist: {}, kruskal p: {}'.format(dist, kruskal(group_level_dists[0],
			group_level_dists[1],
			group_level_dists[2],
			group_level_dists[3],
			group_level_dists[4])[1]))
		assert 	kruskal(group_level_dists[0],group_level_dists[1])[1]<=0.05






