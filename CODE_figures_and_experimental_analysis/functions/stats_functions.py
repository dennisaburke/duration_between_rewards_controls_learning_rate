# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

HELPER FUNCTIONS
"""
import scipy.stats as stats
import pingouin
import pandas as pd
import numpy as np
from pathlib import Path
import os

import lick_photo_functions as lpf
def write_stats_to_excel():
    pass

def t_test_from_dict_list_or_df(data,
                                conditions_to_test,
                                two_tailed = True,
                                assume_equal_var = False,
                                paired = False,
                                one_sample = None,
                                data_is_nested_dict = False,
                                data_is_regular_dict = False,
                                data_is_df = False, ):
    if one_sample is not None:
        if len(conditions_to_test)>1:
            raise Exception('Unclear whether to do 1 or 2-sample test, >1 conditions given with one sample value')
    if (not (data_is_df or data_is_regular_dict or data_is_nested_dict)):
        raise Exception('Unclear what format data is in, check data_is_df, data_is_regular_dict, or data_is_nested_dict flags')
    elif data_is_nested_dict:
        data_conditions = {x: list(data[x].values()) for x in conditions_to_test}

    elif data_is_regular_dict:
        data_conditions = data

    elif data_is_df:
        data_conditions = {x: list(data.loc[x,:]) for x in data.index.get_level_values('condition')}
    t_test = stats.ttest_ind(a = data_conditions[conditions_to_test[0]],
                             b = data_conditions[conditions_to_test[1]],
                             equal_var = assume_equal_var)
    t_test_pingouin = pingouin.ttest(x = data_conditions[conditions_to_test[0]],
                                     y = data_conditions[conditions_to_test[1]],
                                     correction = (not assume_equal_var),
                                     )
    t_test_pingouin['scipy_stats_T'] = t_test[0]
    t_test_pingouin['scipy_stats_pval'] = t_test[1]
    return {'p': t_test[1], 't': t_test[0], 'pingouin':t_test_pingouin}

def convert_dict_to_long_df(data_dict,
                            label_key = '',
                            label_values = '',
                            keys_to_include = 'all',
                            dropna = True,
                            data_is_nested_dict = False,
                            data_is_regular_dict = True,
                            ):
    if keys_to_include == 'all':
        keys_to_keep = list(data_dict.keys())
    else:
        if isinstance(keys_to_include, list):
            keys_to_keep = keys_to_include
        else:
            keys_to_keep = [keys_to_include]
    data_dict = {k: data_dict[k] for k in keys_to_keep }
    if (data_is_nested_dict + data_is_regular_dict) != 1:
        raise Exception('Unclear what format data is in, check data_is_regular_dict and data_is_nested_dict flags')
    elif data_is_nested_dict:
        data = {x: list(data_dict[x].values()) for x in data_dict}
    elif data_is_regular_dict:
        data = data_dict

    df_long = (pd.DataFrame.from_dict(data, orient = 'index')
                                    .rename_axis(label_key).reset_index()
                                    .melt(id_vars = [label_key],
                                          value_name = label_values)
                                    .drop('variable', axis = 1)
                                    .dropna()
                                    )
    return df_long

def one_way_anova_from_dict(data_dict,
                            label_key = '',
                            label_values = '',
                            keys_to_include = 'all',
                            dropna = True,
                            data_is_nested_dict = True,
                            data_is_regular_dict = False,
                            assume_equal_variance = False,
                            save_stats = False,
                            fig_path = ''):
    data_df = convert_dict_to_long_df(data_dict = data_dict,
                                    label_key = label_key,
                                    label_values = label_values,
                                    keys_to_include = keys_to_include,
                                    dropna = dropna,
                                    data_is_nested_dict = data_is_nested_dict,
                                    data_is_regular_dict = data_is_regular_dict,
                                    )
    if assume_equal_variance:
        test = 'ANOVA'
        anova_results = pingouin.anova(dv = label_values,
                                             between = label_key,
                                             data = data_df,
                                             )

    else:
        test = 'Welch ANOVA'
        anova_results = pingouin.welch_anova(dv = label_values,
                                             between = label_key,
                                             data = data_df,
                                             )
    #run same test with scipy to compare if neccessary
    if data_is_nested_dict:
        data_lists  = {x: [y
                        for y
                        in data_dict[x].values()
                        if type(y) is not list
                        ]
                    for x
                    in data_dict
                    }
    elif data_is_regular_dict:
        data_lists = data_dict
    anova_results_scipy = stats.f_oneway(*data_lists.values(), equal_var = assume_equal_variance)
    anova_results['scipy_stats_F'] = anova_results_scipy.statistic
    anova_results['scipy_stats_p'] = anova_results_scipy.pvalue
    if save_stats:
        new_title = lpf.cleanStringForFilename(f'{test}_{data_df[label_key].unique()}_{label_values}')
        stats_fig_path = os.path.join(fig_path, 'stats')
        print(fig_path)
        Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
        anova_results.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}'+'.csv'))
    print((f"{test} results comparing {data_df[label_key].unique()} {label_values}:"
          f"\nddof:{anova_results.ddof1.iloc[0]},{np.round(anova_results.ddof2.iloc[0], 3)}; F = {anova_results.F.iloc[0]}; p = {anova_results['p-unc'].iloc[0]}") )
    return anova_results

def compare_linregress_slope_onesample(linregress_output,
                                        n_observations,
                                        hypothesized_slope = -1,
                                        ):
    t_statistic = (linregress_output.slope - hypothesized_slope) / linregress_output.stderr
    # Degrees of freedom
    doff = n_observations - 2
    p_value_two_tailed = 2 * (1 - stats.t.cdf(np.abs(t_statistic), doff))
    return p_value_two_tailed, t_statistic

def compare_two_linregress_slopes(control, n_control, comparison, n_comparison, two_tailed = True):
    control_slope = control.slope
    control_err = control.stderr
    comparison_slope = comparison.slope
    comparison_err = comparison.stderr
    t_statistic = (control_slope - comparison_slope) / (np.sqrt(control_err**2 + comparison_err**2))
    doff = n_control + n_comparison - 4  # (n1 + n2 - 2*number_of_groups)
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), doff))
    return p_value, t_statistic

def FTest(x, y):
    """
    Calculate the F statistic and test for equal variances
    between two groups (sig result means unequal variances)

    test only works when larger variance is first group
    so check for that first

    two-tailed

    """
    if isinstance(x, pd.Series):
        var_x = x.var(ddof=1)
    else:
        var_x = np.nanvar(x, ddof =1)
    if isinstance(y, pd.Series):
        var_y = y.var(ddof=1)
    else:
        var_y = np.nanvar(y, ddof=1)
    # print(var_x)
    # print(var_y)
    if var_x > var_y:
        group1 = x
        var_group1 = var_x

        group2 = y
        var_group2 = var_y
    elif var_y > var_x:
        group1 = y
        var_group1 = var_y
        group2 = x
        var_group2 = var_x
    elif var_x == var_y:
        f = 1.0
        p = 1.0
        return {'f_stat':f, 'p_val': p, 'var_x': var_x, 'var_y': var_y}

    ddof1 = np.count_nonzero(~np.isnan(group1)) - 1
    ddof2 = np.count_nonzero(~np.isnan(group2)) - 1
    # print(ddof1)
    # print(ddof2)
    f = var_group1 / var_group2

    p = 2.0*(1.0 - stats.f.cdf(f, ddof1, ddof2))
    return {'f_stat':f, 'p_val': p, 'var_x': var_x, 'var_y': var_y, 'ddof1':ddof1, 'ddof2':ddof2}



#def welch_anova_from_dict(data_dict, )

# #%%
# f_test = FTest(x = data_list[0], y = data_list[1])
# t_test = stats.ttest_ind(a = data_list[0], b =data_list[1], equal_var = stats_assume_equal_var)
# t_test_pingouin = pingouin.ttest(x = data_list[0], y = data_list[1], correction = True)
# t_test_pingouin['scipy_stats_T'] = t_test[0]
# t_test_pingouin['scipy_stats_pval'] = t_test[1]
# t_test_pingouin['FTest_F'] = f_test['f_stat']
# t_test_pingouin['FTest_pval'] = f_test['p_val']
# t_test_pingouin['FTest_ddof1'] = f_test['ddof1']
# t_test_pingouin['FTest_ddof2'] = f_test['ddof2']
# t_test_pingouin[f'mean_{conditions[0]}'] = np.nanmean(data_conditions[conditions[0]])
# t_test_pingouin[f'sem_{conditions[0]}'] =stats.sem(data_conditions[conditions[0]], ddof = 1, nan_policy = 'omit')
# t_test_pingouin[f'var_{conditions[0]}'] = np.nanvar(data_conditions[conditions[0]], ddof = 1)
# t_test_pingouin[f'mean_{conditions[1]}'] = np.nanmean(data_conditions[conditions[1]])
# t_test_pingouin[f'sem_{conditions[1]}'] =stats.sem(data_conditions[conditions[1]], ddof = 1, nan_policy = 'omit')
# t_test_pingouin[f'var_{conditions[1]}'] = np.nanvar(data_conditions[conditions[1]], ddof = 1)
# if plot_stats:
#     title = f'mean: p = {np.round(t_test[1], 4)} (t= {round(t_test[0], 4)})\nvariance: p = {round(f_test["p_val"], 4)} (F= {round(f_test["f_stat"], 4)})'
#     bar_ax.set_title(title)
#     ylim_upper = bar_ax.get_ylim()[1]
#     bar_ax.plot([0,1], [ylim_upper,ylim_upper], color = 'k', linewidth = linewidth_stats_lines, markersize = 0)
# if save_stats:
#     stats_fig_path = os.path.join(fig_path, 'stats')
#     Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
#     t_test_pingouin.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}_'+ '_'.join(conditions) +'.csv'))

# elif num_bars > 2:
# pass
# if stats_assume_equal_var:
#     anova_pingouin = pingouin.anova(dv = 'data', between = 'condition', data = anova_df)
#     anova_scipy = stats.f_oneway(*data_list)
#     anova_pingouin['scipy_stats_T'] = anova_scipy[0]
#     anova_pingouin['scipy_stats_pval'] = anova_scipy[1]
# else:
#     anova_pingouin = pingouin.welch_anova(dv = 'data', between = 'condition', data = anova_df)



# for condition in conditions:
#     anova_pingouin[f'mean_{condition}'] = np.nanmean(data_conditions[condition])
#     anova_pingouin[f'sem_{condition}'] =stats.sem(data_conditions[condition], ddof = 1, nan_policy = 'omit')
#     anova_pingouin[f'var_{condition}'] = np.nanvar(data_conditions[condition], ddof = 1)

# if plot_stats:
#     title = f"mean: p = {np.round(anova_pingouin['p-unc'].iloc[0], 4)} (F= {round(anova_pingouin.F.iloc[0], 4)})"
#     bar_ax.set_title(title)
#     ylim_upper = bar_ax.get_ylim()[1]
#     bar_ax.plot([0, len(conditions)-1], [ylim_upper,ylim_upper], color = 'k', linewidth = linewidth_stats_lines, markersize = 0)
# if save_stats:
#     stats_fig_path =os.path.join(fig_path, 'stats')
#     Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
#     anova_pingouin.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}_'+ '_'.join(conditions) +'.csv'))

# #%%