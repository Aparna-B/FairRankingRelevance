import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.special import softmax
import math
import matplotlib
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(6, 3)


colors = ['blue', 'darkorange']


def get_dataset_name(name):
	if 'normal' in name:
		return 'Synthetic-normal'
	elif 'pareto' in name:
		return 'Synthetic-pareto'
	else:
		return 'JEE'


def maxAbsDiff(arr, n):

	# To store the minimum and the maximum
	# elements from the array
	if n == 1:
		return arr[0]

	minEle = arr[0]
	maxEle = arr[0]
	for i in range(1, n):
		minEle = min(minEle, arr[i])
		maxEle = max(maxEle, arr[i])
	return (maxEle - minEle)


def get_unfairness_geometric_attention(df_name, p=0.68, k=10,
	fairness_metric='demographic_fairness', use_geom='Log',
	worthiness='pred',
	imb_ratio=0.5):
	"""
	"""
	df = pd.read_csv(df_name)

	if 'normal' in df_name:
		test_file = '../../example/synthetic_imbalance/synthetic{}_test_normal.csv'.format(
			imb_ratio)
		score_col = 'U'
		sensitive_col = 'G'
		ranked_df = pd.read_csv(test_file)
	elif 'pareto' in df_name:
		test_file = '../../example/synthetic_imbalance/synthetic{}_test_pareto.csv'.format(
			imb_ratio)
		score_col = 'U'
		sensitive_col = 'G'
		ranked_df = pd.read_csv(test_file)
	else:
		raise NotImplementedError

	ranked_df['pred'] = df.values[0, :]
	ranked_df['pred']=softmax(ranked_df.pred.values.reshape(-1, 1))
	ranked_df[score_col]=MinMaxScaler().fit_transform(ranked_df[score_col].values.reshape(-1,1))

	if use_geom=='Geom':
		loc_weights = [p*pow(1-p, i-1) for i in range(k+1)[1:]]
	elif use_geom=='Log':
		loc_weights = [1/math.log2(i+1) for i in range(k+1)[1:]]
	else:
		loc_weights = [1 for i in range(k+1)[1:]]
	loc_weights = np.array(loc_weights)
	loc_weights = loc_weights/np.sum(loc_weights)
	ranked_df = ranked_df.sort_values('pred', ascending=False)
	ranked_df['pos'] = np.arange(ranked_df.shape[0])

	exposure_ratios = []
	cum_exposure = []
	assert ranked_df[sensitive_col].nunique() == 2

	group_names=np.sort(ranked_df[sensitive_col].unique())
	group_names=group_names.tolist()
	group_names.reverse()
	rels=[]
	for g in group_names:
		idx_there = ranked_df[ranked_df[sensitive_col] == g]['pos'].values
		idx_there = idx_there[idx_there < k]
		weights = loc_weights[idx_there]

		if len(weights) == 0:
			curr_exposure = 0
		else:
			curr_exposure = np.sum(weights)
		if worthiness == 'pred':
			curr_rel = ranked_df[ranked_df[sensitive_col] == g]['pred'].sum()
			rels.append(ranked_df[ranked_df[sensitive_col] == g]['pred'].mean())
		else:
			curr_rel = ranked_df[ranked_df[sensitive_col] == g][score_col].sum()
			rels.append(ranked_df[ranked_df[sensitive_col] == g][score_col].mean())

		cum_exposure.append(curr_exposure)
		exposure_ratios.append(curr_exposure/curr_rel)
		
	demographic_fairness = np.abs(cum_exposure[1]/cum_exposure[0])
	exposure_fairness = np.abs(exposure_ratios[1]/exposure_ratios[0])
	print(dist,rels[0]/rels[1],cum_exposure)

	if fairness_metric == 'demographic_fairness':
		return NotImplementedError
	else:
		return exposure_fairness


# defining the parameters for output
p = 0.68
k = 10
fairness_metric = 'exposure_fairness'


seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
imb_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(6, 3)

count = 0
for dist in ['pareto','normal']:
	print(imb_ratios)
	overall_fairness = []
	overall_fairness_sd = []
	diff_fairness = []
	diff_fairness_sd=[]
	for imb_ratio in imb_ratios:
		all_fairness = []
		for worthiness in ['pred', 'not pred']:
			dataset = '../output_replicated/pred_rel/matched_test{}_{}_{}.csv'
			fairness_curr = []
			for seed in seeds:
				fairness_score = get_unfairness_geometric_attention(
					dataset.format(imb_ratio, dist, seed),
					p=p, k=k,
					fairness_metric=fairness_metric,
					worthiness=worthiness,
					imb_ratio=imb_ratio
					)
				fairness_curr.append(fairness_score)
			all_fairness.append(fairness_curr)
		try:
			assert np.nanmean(np.array(all_fairness[0]))>=1
			assert np.nanmean(np.array(all_fairness[1]))>=1
		except:
			# Checking if fairness values are greater than 1. This exception is triggered for one
			# imbalance ratio with normal.
			print(np.nanmean(np.array(all_fairness[0])),np.nanmean(np.array(all_fairness[1])),dist)
		diff_fairness.append(np.nanmean(np.array(all_fairness[0])-np.array(all_fairness[1])))
		diff_fairness_sd.append(np.nanstd(np.array(all_fairness[0])-np.array(all_fairness[1])))
	print(np.min(diff_fairness),np.max(diff_fairness))
	plt.plot(imb_ratios,diff_fairness)
	plt.errorbar(x=imb_ratios,y=diff_fairness,yerr=diff_fairness_sd,
		     fmt='o', color='black',
             ecolor='lightgray')
	plt.xlabel('Imbalance Ratio')
	plt.ylabel('Exposure Fairness: Difference')
	# plt.ylim([-0.4,0.6])
	plt.xlim([0.45,0.95])
	plt.tight_layout()
	plt.savefig('diff_imbalance_fair_{}.png'.format(dist),dpi=300)
	plt.close()


