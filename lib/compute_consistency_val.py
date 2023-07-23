import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,pointbiserialr
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_rel
import matplotlib
import seaborn as sns
from scipy.special import softmax
fig = matplotlib.pyplot.gcf()
FS=19
# fig.set_size_inches(6.2,3)


seeds = [1,2,3,4,5,6,7,8,9,10]
N=5000

for dist in ['normal','pareto','fairtrecnotpretrained']:

	dataset_files=['../output_replicated/pred_rel/val_{}_matched_{}_{}.csv']
	corr_arr=[]
	corr_arr_f=[]
	corr_arr_m=[]

	df_seed=[]
	all_seeds_consistency = []

	for seed in seeds:
		# print('here')
		df_final = pd.read_csv(dataset_files[0].format(str(500),dist,
		seed),usecols=np.arange(N))
		# print(seed)
		consistency_values=[]
		for epoch in np.arange(10, 499, 10):
			# print(epoch)
			df_curr=pd.read_csv(dataset_files[0].format(str(epoch),dist,seed),usecols=np.arange(N))
			# scaling within 0-1 within each run
			diff=df_final.values[0][:N]-df_curr.values[0][:N]
			#NB: Results are mostly same, except for pareto
			# diff=softmax(df_final.values[0][:N])-softmax(df_curr.values[0][:N])

			
			# this is because the validation list also contains the padded values of train
			# assert diff.shape[0]==N #and df_curr.values[0][5000]==df_curr.values[0][5001] and df_final.values[0][5000]==df_final.values[0][5001]
			total_sq = [i**2 for i in diff]
			total_mean = np.mean(total_sq)
			consistency_values.append(total_mean)
		all_seeds_consistency.append(consistency_values)
	all_seeds_consistency = np.mean(all_seeds_consistency, axis=0)
	# assert all_seeds_consistency.shape[0]==np.arange(10, 499, 10).shape[0]
	print(all_seeds_consistency[-2]-all_seeds_consistency[-1])
	plt.plot(np.arange(10, 499, 10),all_seeds_consistency)
	plt.xlabel('Train iteration',fontsize=FS)
	plt.ylabel('MSE from final value',fontsize=FS)
	plt.xticks(fontsize=FS)
	plt.yticks(fontsize=FS)
	# plt.ylim([0.005,0.4])
	plt.tight_layout()
	plt.savefig('val_consistency_{}_plot_unnormalized.png'.format(dist),dpi=300,bbox_inches="tight")
	plt.close()








