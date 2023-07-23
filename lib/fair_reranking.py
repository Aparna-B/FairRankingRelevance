import math
import pandas as pd 
from sklearn.metrics import ndcg_score
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
import numpy as np
import reranking
np.random.seed(1)


def get_unfairness_geometric_attention(df, more_sensitive, less_sensitive,
    p=0.68,k=10,
    fairness_metric='exposure_fairness',use_geom="Log", 
    worthiness='pred',dsname='normal',rel_dict=None,group_count_dict=None):
    """
    """

    if 'normal' in dsname:
        test_file = '../../example/synthetic/synthetic_test_normal.csv'
        score_col = 'U'
        sensitive_col = 'G'
        ranked_df = pd.read_csv(test_file)
    elif 'pareto' in dsname:
        test_file = '../../example/synthetic/synthetic_test_pareto.csv'
        score_col = 'U'
        sensitive_col = 'G'
        ranked_df = pd.read_csv(test_file)
    else:
        test_file = '../../example/fairtrecnotpretrained/fairtrecnotpretrained_test.csv'
        score_col = 'relevance'
        sensitive_col = 'country'
        ranked_df = pd.read_csv(test_file)
        # ranked_df.loc[ranked_df[score_col]>2,'U']=1
        # ranked_df.loc[ranked_df[score_col]<=2,'U']=0

    ranked_df['ind']=np.arange(ranked_df.shape[0])
    ranked_df=ranked_df.merge(df, on='ind')
    ranked_df=ranked_df.sort_values('pred',ascending=False)
    if use_geom=='Geom':
        loc_weights=[p*pow(1-p, i-1) for i in range(k+1)[1:]]
    elif use_geom=='Log':
        loc_weights=[1/math.log2(i+1) for i in range(k+1)[1:]]  
    else:
        loc_weights=[1 for i in range(k+1)[1:]]  
    loc_weights = np.array(loc_weights)
    loc_weights = loc_weights/np.sum(loc_weights)
    ranked_df=ranked_df.sort_values('pos')
    # ranked_df['pos']=np.arange(ranked_df.shape[0])

    exposure_ratios = []
    cum_exposure=[]
    try:
        assert ranked_df[sensitive_col].nunique()==2
    except AssertionError:
        # NB: this means that top-k positions only had one group.
        return 0

    for g in [more_sensitive, less_sensitive]:
        # For fairness interventions, check
        assert rel_dict

        n_group = group_count_dict[g]
        idx_there=ranked_df[ranked_df[sensitive_col]==g]['pos'].values
        idx_there=idx_there[idx_there<k]
        weights=loc_weights[idx_there]
        
        if len(weights)==0:
            curr_exposure=0
        else:
            curr_exposure=np.sum(weights)/n_group


        if rel_dict:
            curr_rel = rel_dict[g]
        else:
            raise NotImplementedError
        cum_exposure.append(curr_exposure)
        exposure_ratios.append(curr_exposure/curr_rel)
    demographic_fairness =  np.abs(cum_exposure[0]/cum_exposure[1])
    exposure_fairness =  np.abs(exposure_ratios[0]/exposure_ratios[1])

    if fairness_metric == 'demographic_fairness':
        return NotImplementedError
    else:
        return exposure_fairness


    
dist='normal'
pre_fair=[]
post_fair=[]
seeds=[1,2,3,4,5,6,7,8,9,10]


# fairness computed using true relevance, obtained from get_group_fairness.py
true_fair={}
true_fair['normal']=[3.34570909513476, 3.34570909513476, 3.34570909513476, 4.318727350235828, 3.34570909513476, 3.3968740254828544, 3.168128141248116, 3.168128141248116, 3.34570909513476, 3.4225669575154]
true_fair['pareto']=[2.6121780591416446, 2.561296365609836, 4.513129238590662, 2.4171534221635325, 1.8046322850295438, 2.4171534221635325, 0.8177486282967175, 2.132247373155067, 1.7839938695086952, 0.7434862171089116]
true_fair['fairtrecnotpretrained']=[0.8407041226424052, 0.0, 0.0, 3.2213778651860876, 0.8407041226424052, 0.0, 0.9927604317496005, 0.0, 2.228447173148185, 1.2946445564396238]
del_indices=[]
for seed in seeds:
    k = 10 # number of topK elements returned (value should be between 10 and 400)

    if dist in ['normal','pareto']:
        test_file='../../example/synthetic/synthetic_test_{}.csv'.format(dist)
        pred_file='../output_replicated/pred_rel/matched_test_{}_{}.csv'.format(dist,seed)

        sensitive='G'
        utility='U'
        more_sensitive='F'
        less_sensitive='M'
    else:
        test_file='../../example/fairtrecnotpretrained/fairtrecnotpretrained_test.csv'.format(dist)
        pred_file='../output_replicated/pred_rel/matched_test_{}_{}.csv'.format(dist,seed)

        sensitive='country'
        utility='relevance'
        more_sensitive=1
        less_sensitive=0


    

    df=pd.read_csv(test_file)
    df_pred = pd.read_csv(pred_file)
    df['pred']=df_pred.values.T
    df['ind']=np.arange(df.shape[0])
    df=df.sort_values('pred',ascending=False)
    df['pred']=softmax(df['pred'].values)
    #df['pred']=df['pred'].values/df['pred'].sum()
    assert df.shape[0]==10000
    prop_dict={group:df[df[sensitive]==group]['pred'].mean() for group in [more_sensitive,less_sensitive]}
    # NB: for later use in computing fairness
    rel_dict={group:df[df[sensitive]==group]['pred'].mean() for group in [more_sensitive,less_sensitive]}
    group_count_dict={group:df[df[sensitive]==group].shape[0] for group in [more_sensitive,less_sensitive]}

    norm_const=np.sum(list(prop_dict.values()))

    for key, value in prop_dict.items():
        prop_dict[key]=value/norm_const

    df['pos']=np.arange(df.shape[0])
    p_before=get_unfairness_geometric_attention(df[['ind','pos','pred']],
        more_sensitive=more_sensitive,
        less_sensitive=less_sensitive,
        dsname=dist,
        rel_dict=rel_dict,group_count_dict=group_count_dict)
    pre_fair.append(p_before)

    # getting top k
    df=df.iloc[0:k]

    item_attribute = df[sensitive].values
    print(prop_dict)

    try:
        rerank_indices = reranking.rerank(
            item_attribute,  # attributes of the ranked items
            prop_dict,  # desired item distribution
            max_na=None,  # controls the max number of attribute categories applied
            k_max=None,  # length of output, if None, k_max is the length of `item_attribute`
            algorithm="det_const_sort",  # "det_greedy", "det_cons", "det_relaxed", "det_const_sort"
            verbose=False,  # if True, the output is with detailed information
        )
        df_reranked=df.iloc[rerank_indices]
        df_reranked['pos']=np.arange(df_reranked.shape[0])
        p_after=get_unfairness_geometric_attention(df_reranked[['ind','pred','pos']],
            more_sensitive=more_sensitive,
            less_sensitive=less_sensitive,
            dsname=dist,
            rel_dict=rel_dict,group_count_dict=group_count_dict)
        post_fair.append(p_after)
        print(df.groupby(sensitive).count())

    except:
        # print(df.groupby(sensitive).count())
        print('Not possible for seed: ',seed)
        del_indices.append(seed-1)
print(pre_fair, post_fair)
print('Before fairness: {}, after fairness: {}'.format(
    np.mean(pre_fair), np.mean(post_fair)))

print('Before fairness_sd: {}, after fairness_sd: {}'.format(
    np.std(pre_fair), np.std(post_fair)))
all_indices=set(np.arange(10).tolist())-set(del_indices)
all_indices=np.array(list(all_indices))

x=np.array(true_fair[dist])
x=np.array(x)
print(wilcoxon(post_fair,x[all_indices]),np.mean(true_fair[dist]))


