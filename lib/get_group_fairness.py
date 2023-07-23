import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr,ttest_rel,kruskal,wilcoxon
from scipy.special import softmax
import math
import matplotlib
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(6,3)


colors=['blue','darkorange']


def get_unfairness_geometric_attention(df_name, p=0.68,k=10,
    fairness_metric='demographic_fairness',use_geom="Log", 
    worthiness='pred'):
    """
    """
    df = pd.read_csv(df_name)

    if 'normal' in df_name:
        test_file = '../../example/synthetic/synthetic_test_normal.csv'
        score_col = 'U'
        sensitive_col = 'G'
        ranked_df = pd.read_csv(test_file)
    elif 'pareto' in df_name:
        test_file = '../../example/synthetic/synthetic_test_pareto.csv'
        score_col = 'U'
        sensitive_col = 'G'
        ranked_df = pd.read_csv(test_file)

    elif 'fairtrecnotpretrained' in df_name:
        test_file = '../../example/fairtrecnotpretrained/fairtrecnotpretrained_test.csv'
        score_col = 'relevance'
        sensitive_col = 'country'
        ranked_df = pd.read_csv(test_file)

    else:
        raise NotImplementedError
    
    ranked_df['pred']=df.values[0,:]
    ranked_df['pred']=softmax(ranked_df.pred.values)
    # NB: Verified that scaling in this manner don't cause the results to change
    # ranked_df['pred']=MinMaxScaler().fit_transform(ranked_df['pred'].values.reshape(-1,1))
    ranked_df[score_col]=MinMaxScaler().fit_transform(ranked_df[score_col].values.reshape(-1,1))
    # NB: Because this is a normalizing term that gets cancelled out, verified that results
    # don't change. 
    # ranked_df[score_col]=ranked_df[score_col]/ranked_df[score_col].sum()

    if use_geom=='Geom':
        loc_weights=[p*pow(1-p, i-1) for i in range(k+1)[1:]]
    elif use_geom=='Log':
        loc_weights=[1/math.log2(i+1) for i in range(k+1)[1:]]  
    else:
        loc_weights=[1 for i in range(k+1)[1:]]  
    loc_weights = np.array(loc_weights)
    loc_weights = loc_weights/np.sum(loc_weights)
    ranked_df=ranked_df.sort_values('pred',ascending=False)
    ranked_df['pred']=ranked_df.pred.values/ranked_df.pred.sum()
    ranked_df['pos']=np.arange(ranked_df.shape[0])

    exposure_ratios = []
    cum_exposure=[]
    assert ranked_df[sensitive_col].nunique()==2
    print(np.sort(ranked_df[sensitive_col].unique()),ranked_df['pred'].min(),ranked_df['pred'].max())


    # NB: to standardize that minority/majority in fairness calc
    if 'fairtrec' in df_name:
        ranked_df[sensitive_col]=1-ranked_df[sensitive_col].values

    for g in np.sort(ranked_df[sensitive_col].unique()):
        idx_there=ranked_df[ranked_df[sensitive_col]==g]['pos'].values
        idx_there=idx_there[idx_there<k]
        weights=loc_weights[idx_there]
        
        if len(weights)==0:
            curr_exposure=0
        else:
            curr_exposure=np.sum(weights) 
            
        if worthiness=='pred':
            curr_rel = ranked_df[ranked_df[sensitive_col]==g]['pred'].sum() 
        else:
            curr_rel = ranked_df[ranked_df[sensitive_col]==g][score_col].sum() 
        cum_exposure.append(curr_exposure)
        exposure_ratios.append(curr_exposure/curr_rel)
    exposure_fairness =  np.abs(exposure_ratios[0]/exposure_ratios[1])

    if fairness_metric == 'demographic_fairness':
        raise NotImplementedError
    else:
        return exposure_fairness





# defining the parameters for output
p=0.68
k=10
fairness_metric = 'exposure_fairness'
worthiness='pred'


seeds=[1,2,3,4,5,6,7,8,9,10]




import matplotlib
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(6,3)

count=0
for dist in ['normal','pareto','fairtrecnotpretrained']:
    overall_fairness=[]
    overall_fairness_sd=[]
    dataset='../output_replicated/pred_rel/matched_test_{}_{}.csv'
    for worthiness in ['pred','not pred']:
        fairness_curr=[]
        for seed in seeds:
            fairness_score=get_unfairness_geometric_attention(dataset.format(dist, 
                seed), 
                p=p,k=k,
                fairness_metric=fairness_metric,
                worthiness=worthiness)
            fairness_curr.append(fairness_score)
        overall_fairness.append(fairness_curr)  
    print(np.mean(overall_fairness[0]),np.mean(overall_fairness[1]))
    print(np.std(overall_fairness[0]),np.std(overall_fairness[1]))
    print(dist, wilcoxon(overall_fairness[0],overall_fairness[1]))
    print(dist, overall_fairness[1])



