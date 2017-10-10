import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from scipy.stats import spearmanr
from vs_metrics import compute_docking_power, compute_screening_power
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import timeit
import random
#-------------------------------------------------------------------------------
def train_test_model(task, sf_name, tr_d, ts_d, model_params):
    pred_task = 'classification' if sf_name == 'bt-screen' else 'regression'
    tr_d = shuffle_data(tr_d, sf_name)
    tr_grp_id = tr_d['grp_ids'].values
    tr_clstr_id = tr_d['clstr_ids'].values
    tr_x_df = tr_d.drop(['label', 'grp_ids', 'clstr_ids', 'ba'], axis=1)
    tr_x = tr_x_df.values
    ftr_names = tr_x_df.columns.tolist()
    tr_y = tr_d['label'].values
    ts_grp_id = ts_d['grp_ids'].values
    ts_clstr_id = ts_d['clstr_ids'].values
    predictions_df = ts_d[['grp_ids', 'label']].copy()
    ts_x = ts_d.drop(['label', 'grp_ids', 'clstr_ids', 'ba'], axis=1).values
    ts_y = ts_d['label'].values
    start_time = timeit.default_timer()
    if sf_name in ['bt-score', 'bt-dock', 'bt-screen']:
        model = train_xgb(model_params, tr_x, tr_y, pred_task)
        ts_p = test_xgb(model, ts_x)
    elif sf_name == 'rf-score':
        model = train_sk_rf(model_params, tr_x, tr_y)
        ts_p = test_sk_models(model, ts_x)
    elif sf_name == 'x-score':
        model = train_sk_lm(model_params, tr_x, tr_y)
        ts_p = test_sk_models(model, ts_x)
    end_time = timeit.default_timer()
    ts_p_proba = ts_p.copy()
    if pred_task == 'classification' and len(np.unique(ts_p)) > 2:
        ts_p[ts_p>0.5] = 1
        ts_p[ts_p<=.5] = 0
        RpAuc, RsTpr, SdTnr, RmseAcc = cls_metrics(ts_y, ts_p, ts_p_proba)
    else:
        RpAuc, RsTpr, SdTnr, RmseAcc = reg_metrics(ts_y, ts_p)
    
    dsizes = [tr_x.shape[0], ts_x.shape[0], tr_x.shape[1]]
    predictions_df['predicted_label'] = ts_p 
    predictions_df.columns = ['complex_id', 'true_label', 'predicted_label']
    perf_df = None
    if task == 'score':
        performance = [RpAuc, RsTpr, SdTnr, RmseAcc]
        #print('Rp = %.3f, Rs = %.3f, SD = %.3f, RMSE = %.3f'%(RpAuc, RsTpr, SdTnr, RmseAcc))
        perf_df = pd.DataFrame(columns=['N_Training', 'N_Test', 'N_Descriptors', 
                                        'Rp', 'Rs', 'SD', 'RMSE'])
        perf_df.loc[0] =  dsizes + performance
    elif task == 'dock':
        is_decreasing = not is_task_specific(task, sf_name)
        performance = compute_docking_power(ts_grp_id, ts_y, ts_p, 
                                            decreasing=is_decreasing)
        # metrics=c('dock_S21','dock_S22','dock_S23')
        perf_df = pd.DataFrame(columns=['N_Training', 'N_Test', 'N_Descriptors', 
                                     'S21', 'S22', 'S23'])
        perf_df.loc[0] =  dsizes + [performance[1], performance[4], performance[5]]
    elif task == 'screen':
        performance = compute_screening_power(ts_grp_id, true_labels=ts_y, 
                                             pred_labels=ts_p_proba, 
                                             decreasing=True)
        perf_df = pd.DataFrame(columns=['N_Training', 'N_Test', 'N_Descriptors', 
                                  'EF1', 'EF5', 'EF10'])
        perf_df.loc[0] =  dsizes + performance[0:3]
    if perf_df is not None:
        perf_df[['N_Training', 'N_Test', 'N_Descriptors']] = perf_df[['N_Training', 'N_Test', 'N_Descriptors']].astype(int)
    return [predictions_df, perf_df]
#-------------------------------------------------------------------------------
def shuffle_data(train, sfname):
    """
    This function has been written long time ago
    and it is more complex than it needs to be.
    It was part of another pipeline that has
    more features than this mini-project.
    We have not replaced it because we wanted to
    generate the same random samples we used in 
    our paper. Otherwise, we could replace it with
    one line of code.
    """
    mx_tr_size_dic = {'rf-score': 3000, 'x-score': 3000,
                      'bt-score': 3000, 'bt-screen': 20000,
                      'bt-dock': 300000}
    mx_tr_size = mx_tr_size_dic[sfname]
    random.seed(1)
    np.random.seed(1)
    train = train.reindex(np.random.permutation(train.index))
    tr_idxs = np.arange(train.shape[0])
    tr_idxs = np.random.permutation(tr_idxs)

    if train.shape[0] > mx_tr_size:
      tr_idxs = np.random.choice(tr_idxs, size=mx_tr_size, replace=False)
      #train = train.iloc[np.random.choice(train.shape[0], mx_tr_size, replace=False)]
      train = train.iloc[tr_idxs]
    return train
#-------------------------------------------------------------------------------
def is_task_specific(task, sf_name):
    res = False
    if task == 'score' and sf_name in ['bt-score', 'rf-score', 'x-score']:
        res = True
    elif task == 'dock' and sf_name == 'bt-dock':
        res = True
    elif task == 'sceen' and sf_name == 'bt-screen':
        res = True
    return res
#-------------------------------------------------------------------------------
def train_xgb(model_params, tr_x, tr_y, pred_task):
    n_tr_x = tr_x.shape[0]
    subsample_max = 80000.0/n_tr_x
    subsample = min(0.6, subsample_max)
    n_trees = 4000 if tr_x.shape[0] < 50000 else 5000
    dtrain = xgb.DMatrix(tr_x, label=tr_y, missing = -999.0)
    if tr_x.shape[0] > 10000:
        param = {'bst:max_depth':7, 'bst:eta':0.035, # for docking s = .3, eta =.035, m.d. = 5, learning_rate = 0.05
                 'learning_rate':0.03, 'gamma':0, # for scoring s = .6, eta =.025, m.d. = 10, learning_rate = 0.02
                 #'bst:colsample_bytree':0.75,
                 'bst:colsample_bylevel':0.75,
                 'bst:subsample':subsample, 'silent':1, 
                 'nthread':model_params['n_cpus']}
    else:
        param = {'bst:max_depth':10, 'bst:eta':0.035, # for docking s = .3, eta =.035, m.d. = 5, learning_rate = 0.05
                 'learning_rate':0.02, 'gamma':0, # for scoring s = .6, eta =.025, m.d. = 10, learning_rate = 0.02
                 #'bst:colsample_bytree':0.75,
                 'bst:colsample_bylevel':0.75,
                 'bst:subsample':subsample, 'silent':1, 
                 'nthread':model_params['n_cpus']}

    if pred_task == 'regression':
        param['eval_metric'] = 'rmse'
        param['objective'] = 'reg:linear'
    elif pred_task == 'classification':
        param['eval_metric'] = 'auc'
        param['objective'] = 'binary:logistic'
    #print(param)
    model = xgb.train(param, dtrain, num_boost_round=n_trees, verbose_eval=50)
    return model
#-------------------------------------------------------------------------------
def train_sk_lm(model_params, tr_x, tr_y):
    model = LinearRegression()
    model.fit(tr_x, tr_y.ravel())
    return model
#-------------------------------------------------------------------------------
def train_sk_rf(model_params, tr_x, tr_y):
    n_tr_x = tr_x.shape[0]
    subsample_max = 15000.0/n_tr_x
    subsample = min(0.6, subsample_max)
    n_cpus = model_params['n_cpus']
    try:
        # Version 0.18.1 takes in the subsample parameter
        model = RandomForestRegressor(n_estimators=3000, max_depth=None, 
                                      subsample=subsample,
                                      random_state=0, n_jobs=n_cpus, oob_score=False,
                                      verbose=0)
    except:
        # Version 0.19 and higher does not take the subsample parameter
        model = RandomForestRegressor(n_estimators=3000, max_depth=None, 
                              random_state=0, n_jobs=n_cpus, oob_score=False,
                              verbose=0)
    model.fit(tr_x, tr_y.ravel())
    return model
#-------------------------------------------------------------------------------
def test_xgb(model, ts_x):
    ts_x = xgb.DMatrix(ts_x.copy())
    preds = model.predict(ts_x)
    return preds
#-------------------------------------------------------------------------------
def test_sk_models(model, ts_x, proba=False):
    if proba and hasattr(model, 'predict_proba'):
        preds = model.predict_proba(ts_x)[:,1]
    else:
        preds = model.predict(ts_x).flatten()
    return preds
#-------------------------------------------------------------------------------
def reg_metrics(y, p):
    rp = round(np.corrcoef(y, p)[0,1], 3)
    mse = metrics.mean_squared_error(y, p)
    rmse = round(np.sqrt(mse), 3)
    rs = round(spearmanr(y, p)[0], 3) 
    sd = round(np.sqrt(sum((p-y)**2)/(len(y)-2.0)), 3)
    return [rp, rs, sd, rmse]
#-------------------------------------------------------------------------------
def cls_metrics(y, p, p_proba):
    auc = metrics.roc_auc_score(y, p_proba)
    acc = metrics.accuracy_score(y, p)
    tpr = metrics.recall_score(y, p)# a.k.sensitivity = TP/(TP+FN)
    tn, fp, fn, tp = metrics.confusion_matrix(y, p).ravel()
    tnr = tn*1.0/(tn+fp)# a.k.a. speceificity
    return [round(auc, 3), round(tpr, 3), round(tnr, 3), round(acc, 3)]