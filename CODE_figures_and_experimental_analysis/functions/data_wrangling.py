# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

HELPER FUNCTIONS
"""

import pandas as pd

import lick_photo_functions as lpf

def get_average_of_single_day_dff_reward(trial_df,
                                        peak_or_auc ='auc',
                                        DA_time_wind = '_500ms',
                                        day_to_average = 1,
                                        ):
    mean_day_reward_dff = (trial_df[trial_df['day_num']==day_to_average]
                             .groupby(['condition',
                                       'animal'])
                             [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}']
                             .agg(lambda g: g.iloc[0:day_to_average].mean())
                             )
    return mean_day_reward_dff
def sort_dff_reward_responses(trial_df,
                          peak_or_auc ='auc',
                          DA_time_wind = '_500ms',
                          num_to_return = 3,
                          ):
    reward_value_to_sort = [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}']
    reward_sorted_df = group_and_sort_df(trial_df,
                                         value_to_sort = reward_value_to_sort)
    return reward_sorted_df
def sort_dff_cue_responses(trial_df,
                        peak_or_auc ='auc',
                        DA_time_wind = '_500ms',
                        num_to_return = 3,
                        ):
    antic_value_to_sort = [f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}']
    antic_sorted_df = group_and_sort_df(trial_df,
                                         value_to_sort = antic_value_to_sort,
                                         num_to_return = num_to_return)
    return antic_sorted_df
def get_average_of_max_dff_reward_for_omissions_norm(trial_df,
                                                    peak_or_auc ='auc',
                                                    omission_wind = '2',
                                                    num_to_average = 3,
                                                    ):

    reward_sorted_df = sort_dff_reward_responses(trial_df,
                                            peak_or_auc = peak_or_auc,
                                            DA_time_wind = '_'+omission_wind+'s',
                                            num_to_return = num_to_average,
                                            )
    DA_normalization_omission = (reward_sorted_df
                                 .groupby(['condition',
                                           'animal'])
                                 [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned_{omission_wind}s']
                                 .agg(lambda g: g.iloc[0:num_to_average].mean())
                                 )
    return DA_normalization_omission


def group_and_sort_df(trial_df,
                      value_to_sort = ['epoch_dff_auc_consume_norm_lickaligned_500ms'],
                      group_vars = ['condition',
                                  'animal'],
                      num_to_return = 3,
                      ascending=False,
                      ):
    sorted_df = (trial_df.sort_values(value_to_sort,
                                      ascending=ascending
                                      )
                         .groupby(group_vars)
                         .head(num_to_return)
                         )
    return sorted_df

def get_average_of_max_dff_reward(trial_df,
                              peak_or_auc ='auc',
                              DA_time_wind = '_500ms',
                              num_to_average = 3,
                              ):
    reward_sorted_df = sort_dff_reward_responses(trial_df,
                                            peak_or_auc = peak_or_auc,
                                            DA_time_wind = DA_time_wind,
                                            num_to_return = num_to_average,
                                            )
    average_of_max_rewards = (reward_sorted_df
                        .groupby(['condition','animal'])
                        [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}']
                        .agg(lambda g: g.iloc[0:num_to_average].mean())
                        )
    return average_of_max_rewards

def get_average_of_max_dff_cue(trial_df,
                              peak_or_auc ='auc',
                              DA_time_wind = '_500ms',
                              num_to_average = 3,
                              ):
    antic_sorted_df = sort_dff_cue_responses(trial_df,
                                            peak_or_auc = peak_or_auc,
                                            DA_time_wind = DA_time_wind,
                                            num_to_return = num_to_average,
                                            )
    average_of_max_cue = (antic_sorted_df
                        .groupby(['condition','animal'])
                        [f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}']
                        .agg(lambda g: g.iloc[0:num_to_average].mean())
                        )
    return average_of_max_cue
def get_DA_normalization_cue_reward(trial_df,
                                 norm_to_max_individual_rewards = 3,
                                 norm_to_max_individual_antic = 0,
                                 norm_to_day = 0,
                                 peak_or_auc = 'auc',
                                 DA_time_wind = '_500ms',
                                 ):

    lpf.check_flags(norm_to_max_individual_rewards,
                     norm_to_max_individual_antic,
                     norm_to_day,
                     allow = 'at_most_one',
                     )

    if norm_to_max_individual_rewards:
        DA_normalization = get_average_of_max_dff_reward(trial_df,
                                                      peak_or_auc = peak_or_auc,
                                                      DA_time_wind = DA_time_wind,
                                                      num_to_average = norm_to_max_individual_rewards,
                                                      )
    elif norm_to_max_individual_antic:
        DA_normalization = get_average_of_max_dff_cue(trial_df,
                                                      peak_or_auc = peak_or_auc,
                                                      DA_time_wind = DA_time_wind,
                                                      num_to_average = norm_to_max_individual_rewards,
                                                      )
    elif norm_to_day:


        DA_normalization = get_average_of_single_day_dff_reward(trial_df,
                                                                peak_or_auc = peak_or_auc,
                                                                DA_time_wind = DA_time_wind,
                                                                day_to_average = 1,
                                                                )
    else:
        DA_normalization = 1


    return DA_normalization

def get_DA_normalization_omissions(trial_df,
                                    norm_to_max_individual_rewards = 3,
                                    norm_to_max_individual_antic = 0,
                                    norm_to_day = 0,
                                    DA_normalization_precalced = None,
                                    norm_omission_like_cue = False,
                                    peak_or_auc = 'auc',
                                    DA_time_wind = '_500ms',
                                    omission_wind = '2',
                                    ):

    lpf.check_flags(norm_to_max_individual_rewards,
                     norm_to_max_individual_antic,
                     norm_to_day,
                     allow = 'at_most_one',
                     )
    if norm_to_max_individual_rewards:

        DA_normalization_omission = get_average_of_max_dff_reward_for_omissions_norm(trial_df,
                                                            peak_or_auc = peak_or_auc,
                                                            omission_wind = omission_wind,
                                                            num_to_average = norm_to_max_individual_rewards,
                                                            )
    elif norm_to_max_individual_antic:
        # DA_normalization = get_average_of_max_dff_cue(trial_df,
        #                                               peak_or_auc = peak_or_auc,
        #                                               DA_time_wind = DA_time_wind,
        #                                               num_to_average = norm_to_max_individual_rewards,
        #                                               )
        pass
    elif norm_to_day:


        # DA_normalization = get_average_of_single_day_dff_reward(trial_df,
        #                                                         peak_or_auc = peak_or_auc,
        #                                                         DA_time_wind = DA_time_wind,
        #                                                         day_to_average = 1,
        #                                                         )
        pass
    else:
        DA_normalization_omission = 1

    return DA_normalization_omission

def normalize_DA_cue_reward(da_df,
                                          full_trial_df,
                                            norm_to_max_individual_rewards = 3,
                                            norm_to_max_individual_antic = 0,
                                            norm_to_day = 0,
                                            peak_or_auc = 'auc',
                                            DA_time_wind = '_500ms',
                                            flatten_norm_df = False,
                                            ):
    if sum(bool(f) for f in [norm_to_max_individual_rewards, norm_to_max_individual_antic, norm_to_day]):
        DA_normalization = get_DA_normalization_cue_reward(full_trial_df,
                                                           norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                           norm_to_max_individual_antic = norm_to_max_individual_antic,
                                                           norm_to_day = norm_to_day,
                                                           peak_or_auc = peak_or_auc,
                                                           DA_time_wind = DA_time_wind,
                                                           )
    else:
        DA_normalization = 1
    if flatten_norm_df:
        if isinstance(DA_normalization.index, pd.MultiIndex):
            DA_normalization.index = DA_normalization.index.get_level_values(-1)
    normalized_df = da_df / DA_normalization
    return normalized_df


def get_mean_values_from_trial_range(trial_df,
                                     conditions_to_subset,
                                     range_to_subset,
                                     data_to_return = 'lick',
                                     norm_to_max_individual_rewards = 3,
                                     return_as_df = True,
                                     return_as_nested_dict = False,
                                     return_as_regular_dict = False,
                                     ):
    """

    Parameters
    ----------
    trial_df : TYPE
        DESCRIPTION.
    conditions_to_plot : TYPE
        DESCRIPTION.
    range_to_plot : TYPE
        DESCRIPTION.
    data_to_return : TYPE, optional
        DESCRIPTION. The default is 'lick'.
    norm_to_max_individual_rewards : TYPE, optional
        DESCRIPTION. The default is 3.
     : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    condition_means : TYPE
        DESCRIPTION.

    """
    conditions = lpf.get_conditions_as_list(conditions_to_subset, trial_df)

    df_subset = trial_df[trial_df['condition'].isin(conditions)].copy()
    df_subset_range = pd.DataFrame()
    for cond, wind in range_to_subset.items():
        df_single_condition = df_subset[((df_subset['condition'] == cond)
                                         & (df_subset['cue_trial_num'] <= wind[-1])
                                         & (df_subset['cue_trial_num'] >= wind[0]))]
        df_subset_range = pd.concat([df_subset_range, df_single_condition],
                                     axis = 0, ignore_index=True)

    if data_to_return == 'lick':
        data_to_plot = 'antic_norm_rate_change'
    elif (data_to_return == 'DA'
          or data_to_return == 'da'
          or data_to_return == 'dopamine'):
        data_to_plot = 'epoch_dff_auc_antic_norm_500ms'
    elif data_to_return == 'antic_lick_prob':
        data_to_plot = 'antic_lick_prob'
    else:
        raise Exception('unclear what to plot')

    condition_means = df_subset_range.groupby(['condition','animal',])[data_to_plot].mean()
    if (norm_to_max_individual_rewards
        and data_to_plot == 'epoch_dff_auc_antic_norm_500ms'):
        max_reward_values = trial_df.sort_values(['epoch_dff_auc_consume_norm_lickaligned_500ms'],
                                                  ascending=False).groupby(['animal']).head(10)
        DA_normalization = max_reward_values.groupby(['animal'])['epoch_dff_auc_consume_norm_lickaligned_500ms'].agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())
        condition_means = condition_means/DA_normalization
    if (return_as_df + return_as_nested_dict + return_as_regular_dict) > 1:
        raise Exception('unclear what format to return data in. check return_as_df, return_as_nested_dict, or return_as_regular_dict flags')
    elif return_as_df:
        return condition_means
    elif return_as_nested_dict or return_as_regular_dict:
        means_dict = {}
        for condition in conditions:
            single_con_mean_dict = {condition: condition_means.loc[condition].to_dict()}
            means_dict = {**means_dict, **single_con_mean_dict}
        if return_as_nested_dict:
            return means_dict
        elif return_as_regular_dict:
            return {x: list(means_dict[x].values()) for x in means_dict}
    else:
        raise Exception('unclear what format to return data in. check return_as_df, return_as_nested_dict, or return_as_regular_dict flags')