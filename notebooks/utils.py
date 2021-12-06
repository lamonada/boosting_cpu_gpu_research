import os, sys, json, re, dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from category_encoders import CountEncoder
#from catboost import CatBoostClassifier
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from scipy.stats import ks_2samp as ks

def fit_models(p_dict, r_states, model_lib = 'catboost', fit_type = 'gpu', cat_feats = [],
               x_train=None, y_train=None, x_test=None, y_test=None):
    
    model_names, starts_fit, ends_fit, roc_auc  = [], [], [], []
    for key, value in p_dict.items():
        for random_st in r_states:
            param_dict = value
            param_dict["random_state"] = random_st
            model_name = model_lib + '_' + fit_type + '_' + key + "_" + str(random_st) + "_500K.dill"
            model_names.append(model_name)
            
            if model_lib == 'catboost':
                model = CatBoostClassifier(**param_dict)
            elif model_lib == 'lgbm':
                model = Pipeline(steps=[
                    ("encoder", CountEncoder(cols=cat_feats)),
                    ("estimator", LGBMClassifier(**param_dict))])
                
            else: 
                model = Pipeline(steps=[
                    ("encoder", CountEncoder(cols=cat_feats)),
                    ("estimator", XGBClassifier(**param_dict))])
                
            start_fit = time.time()
            model.fit(x_train, y_train)
            end_fit = time.time()

            starts_fit.append(start_fit)
            ends_fit.append(end_fit)

            roc_auc_score_ = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
            roc_auc.append(roc_auc_score_)
            
            with open('../models/' + model_name, "wb") as f_out:
                dill.dump(model, f_out)
    
#            print('Saved model: ', model_name)
    
    df_res = pd.DataFrame(list(zip(model_names, roc_auc, starts_fit, ends_fit)),
                          columns = ['model_name', 'roc_auc', 'start_fit', 'end_fit'])
    df_res['fitting_time'] = df_res['end_fit'] - df_res['start_fit']
    df_res = df_res.sort_values(by=['model_name'])
    csv_file_name = model_lib + '_' + fit_type + '_500K_metrics__final.csv'
    df_res.to_csv('../metrics/' + csv_file_name, sep=';', index=False)
    print('Metrics df saved to:', '../metrics/' + csv_file_name)
    
    
def get_fitting_time(df_gpu, df_cpu):
    time_gpu_1 = int(df_gpu.loc[0:10, ['fitting_time']].mean())
    time_gpu_2 = int(df_gpu.loc[11:20, ['fitting_time']].mean())
    time_gpu_3 = int(df_gpu.loc[21:30, ['fitting_time']].mean())
    
    time_cpu_1 = int(df_cpu.loc[0:10, ['fitting_time']].mean())
    time_cpu_2 = int(df_cpu.loc[11:20, ['fitting_time']].mean())
    time_cpu_3 = int(df_cpu.loc[21:30, ['fitting_time']].mean())
    
    print('GPU fitting time: ', time_gpu_1, time_gpu_2, time_gpu_3, '\n',
          'CPU fitting time: ', time_cpu_1, time_cpu_2, time_cpu_3, '\n',
          'Difference: ',
          np.round(time_cpu_1/time_gpu_1, 1),
          np.round(time_cpu_2/time_gpu_2, 1),
          np.round(time_cpu_3/time_gpu_3, 1)
         )

    
def get_metrics_with_conf_int(df_test = None, month_list=None, pred_cols = None):
    df_metrics = pd.DataFrame()
    for col in pred_cols:
        auc_list = get_month_metrics(df_test, y_pred_col = col).iloc[1].values[1:]
        df_metrics[col] = auc_list
    df_metrics.index = month_list
    return df_metrics


def get_mean_and_std(df_metrics):
    mean, std, double_std, var = [], [], [], []
    for index in df_metrics.index:
        mean.append(df_metrics[df_metrics.index==index].values.mean())
        var.append(df_metrics[df_metrics.index==index].values.var())
        std.append(df_metrics[df_metrics.index==index].values.std())
        double_std.append(1.96 * df_metrics[df_metrics.index==index].values.std())
        ul = [x-y for x, y in zip(mean, double_std)]
        ol = [x+y for x, y in zip(mean, double_std)]
    return mean, var, std, ul, ol


# функция для подсчета средней корреляции без учета диагональных единиц
def get_corrs_same_params(df, cols):
    corr = df[cols].corr().stack()
    corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)]
    min_, mean_, max_ = corr.min(), corr.mean(), corr.max()
    print('min: ', np.round(min_,3), 'max: ', np.round(max_,3), 'mean: ', np.round(mean_,3))
    

def cpu_gpu_corr_stats(set_1 = None, set_2 = None, set_3 = None):
    set_1_min, set_1_max, set_1_mean = np.diag(set_1).min(), np.diag(set_1).max(), np.diag(set_1).mean()
    set_2_min, set_2_max, set_2_mean = np.diag(set_2).min(), np.diag(set_2).max(), np.diag(set_2).mean()
    set_3_min, set_3_max, set_3_mean = np.diag(set_3).min(), np.diag(set_3).max(), np.diag(set_3).mean()
    print('Param set 1 scores correlations:', 'min:', np.round(set_1_min, 3),
          'max:', np.round(set_1_max, 3), 'mean', np.round(set_1_mean, 3))
    print('Param set 2 scores correlations:', 'min:', np.round(set_2_min, 3),
          'max:', np.round(set_2_max, 3), 'mean', np.round(set_2_mean, 3))
    print('Param set 3 scores correlations:', 'min:', np.round(set_3_min, 3),
          'max:', np.round(set_3_max, 3), 'mean', np.round(set_3_mean, 3))
    

def get_min_corr_models(df_corr):
    min_col_name = df_corr.min().idxmin()
    min_row_idx = df_corr[min_col_name].idxmin()
    min_corr = np.round(df_corr.min().min(), 4)
    return min_col_name, min_row_idx, min_corr


def get_fi_rank_corr(fi_1, fi_2, n_feats=303):
    fi_1['rank_1'], fi_2['rank_2'] = fi_1.index, fi_2.index
    fi_1_ranks = pd.Series(fi_1.rank_1.values, index=fi_1.feature).to_dict()
    fi_2['rank_1'] = fi_2['feature'].map(fi_1_ranks)
    fi_2_ = fi_2.head(n_feats)
    return spearmanr(fi_2_['rank_1'], fi_2_['rank_2'], nan_policy='omit')[0]


def get_param_sets_auc(df, random_seeds, params, model, tr_size):
    """ model in format 'catboost_cpu' """
    auc_list = []
    for param in params:
        for seed in random_seeds:
            model_name = model + '_' + param + '_' + str(seed) + '_' +  tr_size
            auc_list.append(np.round(auc_(df['max_target'], df[model_name]), 6))
    auc_1, auc_2, auc_3 = auc_list[0:10], auc_list[10:20], auc_list[20:30]
    return auc_1, auc_2, auc_3


def get_feat_imp(model, model_type="catb"):
    if model_type == "catb":
        df_res = (
            model
            .get_feature_importance(prettified=True)
            .rename(columns={"Feature Id": "feature", "Importances": "importance"})
        )
    elif model_type == "xgb":
        feat_imp = (
            model.steps[1][1].get_booster().get_score(importance_type="gain")
        )
        keys, values = list(feat_imp.keys()), list(feat_imp.values())
        df_res = pd.DataFrame({"feature": keys, "importance": values}).sort_values(
            by="importance", ascending=False
        )
    else:
        importances = model.steps[1][1].booster_.feature_importance(importance_type='gain')
        features = model.steps[1][1].feature_name_
        df_res = pd.DataFrame(
            {'feature': features, 'importance': importances}).sort_values(
            by = "importance", ascending=False)
        
    # в катбусте индексы отсортированы, в xgboost идут в порядке, как подавались фичи, поэтому:
    df_res = df_res.reset_index(drop=True) 
    return df_res

def get_stat_test_df(df = None, cols_1 = None, cols_2 = None):
    col_names, p_values = [], []
    for col_1 in cols_1:
        for col_2 in cols_2:
            if col_1 != col_2:
                col_names.append(col_1 + '_' + col_2)
                p_value = ks(df[col_1].values, df[col_2].values, alternative='two-sided', mode='auto')[1]
                p_values.append(p_value)
    
    df_res = pd.DataFrame({'models': col_names, 'p_value': p_values})
    print('Cnt models pairs with same scores distribution', df_res[df_res['p_value']>0.05].shape[0])
    print('Cnt models pairs different scores distribution', df_res[df_res['p_value']<=0.05].shape[0])
    return df_res

def get_null_imp_feat_counts(model_names = None, model_type='catb'):
    fi_null_list = []
    if model_type == 'catb':
        for model_name in model_names:
            with open('../models/final_models/' + model_name, 'rb') as f_in:
                model = dill.load(f_in)
            fi = get_feat_imp(model = model, model_type=model_type)
            null_cnt = fi[fi['importance']<0.000001].shape[0]
            fi_null_list.append(null_cnt)
    elif model_type == 'lgbm':
        for model_name in model_names:
            with open('../models/final_models/' + model_name, 'rb') as f_in:
                model = dill.load(f_in)
            fi = get_feat_imp(model = model, model_type=model_type)
            null_cnt = fi[fi['importance']<0.000001].shape[0]
            fi_null_list.append(null_cnt)
    else:
        for model_name in model_names:
            with open('../models/final_models/' + model_name, 'rb') as f_in:
                model = dill.load(f_in)
            fi = get_feat_imp(model = model, model_type=model_type)
            null_cnt = 303 - fi.shape[0]
            fi_null_list.append(null_cnt)        
    return fi_null_list[0:10], fi_null_list[10:20], fi_null_list[20:30]


def get_mean_rank_corr(model_names_1=None, model_names_2=None, model_type='catb'):
    steps = np.arange(10, 301, 10)
    df_rank_corr = pd.DataFrame()
    if model_type == 'catb':
        for model_name_1 in model_names_1:
            for model_name_2 in model_names_2:
                if model_name_1 != model_name_2:
                    col_name = model_name_1 + model_name_2
                    with open('../models/final_models/' + model_name_1, 'rb') as f_in:
                        model_1 = dill.load(f_in)
                    with open('../models/final_models/' + model_name_2, 'rb') as f_in:
                        model_2 = dill.load(f_in)
                    fi_corr_list = []
                    for step in steps:
                        fi_1 = get_feat_imp(model = model_1, model_type=model_type)
                        fi_2 = get_feat_imp(model = model_2, model_type=model_type)
                        fi_corr_list.append(np.round(get_fi_rank_corr(fi_1, fi_2, step), 3))
                    df_rank_corr[col_name] = fi_corr_list
    else:
        for model_name_1 in model_names_1:
            for model_name_2 in model_names_2:
                if model_name_1 != model_name_2:
                    col_name = model_name_1 + model_name_2
                    print(col_name)
                    with open('../models/final_models/' + model_name_1, 'rb') as f_in:
                        model_1 = dill.load(f_in)
                    with open('../models/final_models/' + model_name_2, 'rb') as f_in:
                        model_2 = dill.load(f_in)
                    print('models read')
                    fi_corr_list = []
                    for step in steps:
                        fi_1 = get_feat_imp(model = model_1, model_type=model_type)
                        fi_2 = get_feat_imp(model = model_2, model_type=model_type)
                        print('get imp done')
                        fi_corr_list.append(np.round(get_fi_rank_corr(fi_1, fi_2, step), 3))
                    df_rank_corr[col_name] = fi_corr_list        
    return df_rank_corr


def get_train_test_metrics(df_train, df_oot, gpu_col, cpu_col):
    gpu_auc_train = np.round(auc(df_train['max_target'], df_train[gpu_col]), 3)
    gpu_auc_test = np.round(auc(df_oot['max_target'], df_oot[gpu_col]), 3)
    cpu_auc_train = np.round(auc(df_train['max_target'], df_train[cpu_col]), 3)
    cpu_auc_test = np.round(auc(df_oot['max_target'], df_oot[cpu_col]), 3)
    print("Train / test GPU ROC AUC diff: ", np.round(gpu_auc_train - gpu_auc_test, 3))
    print("Train / test CPU ROC AUC diff: ", np.round(cpu_auc_train - cpu_auc_test, 3))
    print("Train CPU vs GPU ROC AUC diff: ", np.round(cpu_auc_train - gpu_auc_train, 3))
    print("Test CPU vs GPU ROC AUC diff: ", np.round(cpu_auc_test - gpu_auc_test, 3))


def get_month_metrics(df_test=None, y_pred_col=None, target = 'target'):
    
    df_test['date'] = pd.to_datetime(df_test['date'], format='%Y-%m-%d')
    df_test_tmp = df_test[~df_test[target].isnull()]
    month_list = sorted(list(df_test_tmp['month'].unique()))
    month_list.insert(0, 'metrics')
    month_shape_l, roc_auc_l  = ['month_shape'], ['ROC_AUC']
    
    for month in month_list[1:]:
        df_month = df_test_tmp[df_test_tmp['month']==month]
        y_true, y_score = df_month[target], df_month[y_pred_col]
        roc_auc = roc_auc_score(y_score = y_score, y_true = y_true)
        month_shape_l.append(df_month.shape[0])
        roc_auc_l.append(roc_auc)
           
    df_out = pd.DataFrame(columns=month_list)
    df_out = df_out.append(pd.Series(month_shape_l, index=month_list), ignore_index=True)
    df_out = df_out.append(pd.Series(roc_auc_l, index=month_list), ignore_index=True)
    
    return df_out
    

def reduce_memory(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
