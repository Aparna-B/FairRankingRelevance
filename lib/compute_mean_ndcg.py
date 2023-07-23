import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,pointbiserialr
from scipy.stats import ttest_rel
import matplotlib
import seaborn as sns
fig = matplotlib.pyplot.gcf()

seeds=np.arange(1,11)



for dist in ['normal','pareto','fairtrecnotpretrained']:
	df_all=[]
	for seed in seeds:
		csv_file="../output_replicated/matched_test_{}_{}.csv".format(dist,seed)
		df=pd.read_csv(csv_file)
		df_all.append(pd.DataFrame(df))
	df_all=pd.concat(df_all).reset_index()
	df_all=df_all.mean()
	print(dist)
	print(df_all)


for dist in ['normal','pareto']:
	for imb_ratio in [0.5,0.6,0.7,0.8,0.9]:
		df_all=[]
		for seed in seeds:
			csv_file="../output_replicated/matched_test{}_{}_{}.csv".format(imb_ratio,
				dist,seed)
			df=pd.read_csv(csv_file)
			df_all.append(pd.DataFrame(df))
		df_all=pd.concat(df_all).reset_index()
		df_all=df_all.mean()
		print(dist,imb_ratio)
		print(df_all['ndcg_10'])

