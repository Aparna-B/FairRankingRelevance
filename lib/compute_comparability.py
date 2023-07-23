import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,pointbiserialr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import ttest_rel
import matplotlib
import seaborn as sns
from scipy.special import softmax

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(6.2,3)

# all seeds
seeds=np.arange(1,11)


# spearman correlation between scores
spearman_all=[]
spearman_f=[]
spearman_m=[]

rel_true_ratio=[]
rel_pred_ratio=[]
for dist in ['normal','pareto','fairtrecnotpretrained']:
	dataset_files=['../output_replicated/pred_rel/matched_test_{}_{}.csv']
	corr_arr=[]
	corr_arr_f=[]
	corr_arr_m=[]
	seed_level_true_ratio=[]
	seed_level_pred_ratio=[]
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
			sens_col='country'
		else:
			df_all = pd.read_csv('../../example/synthetic/synthetic_test_{}.csv'.format(
				dist))
			score_col='U'
			sens_col='G'

		all_val=df.values[0].squeeze()
		df_all=df_all.iloc[0:len(all_val)]
		df_all['prediction']=softmax(all_val[0:df_all.shape[0]])
		df_all[score_col]=MinMaxScaler().fit_transform(df_all[score_col].values.reshape(-1,1))
		# NB: normalizing the true score by sum (i.e., converting it into a probability) does not cause
		# results to vary.
		df_all['id']=np.arange(df_all.shape[0])
		count_df=df_all.groupby(sens_col).count().reset_index()
		min_count=count_df[count_df.id== count_df.id.min()][sens_col].values[0]
		max_count=count_df[count_df.id== count_df.id.max()][sens_col].values[0]
		df_f=df_all[df_all[sens_col]==min_count]
		df_m=df_all[df_all[sens_col]==max_count]
		corr_all=spearmanr(df_all.prediction.values,df_all[score_col].values)
		corr_f=spearmanr(df_f.prediction.values,df_f[score_col].values)
		corr_m=spearmanr(df_m.prediction.values,df_m[score_col].values)
		corr_arr.append(corr_all)
		corr_arr_f.append(corr_f)
		corr_arr_m.append(corr_m)
		seed_level_true_ratio.append(df_f[score_col].mean()/df_m[score_col].mean())
		seed_level_pred_ratio.append(df_f.prediction.mean()/df_m.prediction.mean())
	spearman_all.append(np.mean(corr_arr))
	spearman_f.append(np.mean(corr_arr_f))
	spearman_m.append(np.mean(corr_arr_m))
	rel_true_ratio.append(np.mean(seed_level_true_ratio))
	rel_pred_ratio.append(np.mean(seed_level_pred_ratio))

print('Considering individual fairness definitions')
for i, dist in enumerate(['normal','pareto','fairtrecnotpretrained']):
	print('Comparability values for dist: {} are - all: {}, F: {}, M: {}'.format(
		dist,
		spearman_all[i],
		spearman_f[i],
		spearman_m[i]))


print('Considering group fairness definitions')
for i, dist in enumerate(['normal','pareto','fairtrecnotpretrained']):
	print('Comparability values for dist: {} are - true: {}, pred: {}'.format(
		dist,
		rel_true_ratio[i],
		rel_pred_ratio[i]))

