# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

EXTENDED DATA FIGURE 8, PANELS F THROUGH K
"""


"""
imports
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import functions.load_preprocess  as lp
import functions.default_configs as dc
import functions.figure_functions as ff
import functions.lick_photo_functions as lpf


"""
set paths to data and outputs
"""

#set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

nwb_dir_path = r'..\DATA_experimental\001632'
figure_path_root = r'..\FIGURES'
fig_path_ExtDataFig8fk = os.path.join(figure_path_root, r'ExtDataFig8fk')

save_figs = True
save_stats = True
#%%
"""
load and prepare data
"""
nwb_file_info_df = lp.get_all_nwb_files_by_condition(nwb_dir_path,  ['30s', '60s', '300s', '600s', '3600s'])
all_trial_data_df, df, all_session_data_df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               return_session_df = True,
                                                               )
df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)

#%%
"""
EXTENDED DATA FIGURE 8
 - CALCULATE ITI LICK RATES AND REWARD CONSUMPTION BOUT DURATION
"""
pre_cue_window = 10
conditions_to_analyze = ['30s',
                         '60s',
                         '300s',
                         '600s',
                         '3600s',
                         ]

num_trials = {'30s': 800,
              '60s': 400,
              '300s': 80,
              '600s': 40,
              '3600s': 8,
              }
first_last_per_day = 6
ITI_lick_rates_dict = dict.fromkeys(conditions_to_analyze, None)
reward_bout_dur_dict = dict.fromkeys(conditions_to_analyze, None)
reward_bout_dur_before_after = dict.fromkeys(conditions_to_analyze, None)
session_df =all_session_data_df[all_session_data_df['condition'].isin(conditions_to_analyze)].copy()
session_df = session_df[session_df['day_num']<=8]
session_df = session_df[session_df['animal'] != '600sD_F7']
animals_by_condition = session_df.groupby(['condition'])['animal'].unique().to_dict()
num_animals_by_condition = {x: len(animals_by_condition[x])
                                for x
                                in animals_by_condition.keys()}
excluded_trials = {}

learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_analyze,

                                                      )
learned_trial_data_lick = learned_trial_data['learned_trial_lick']
prelearning_lick_rates = dict.fromkeys(conditions_to_analyze, {})
ITI_lick_rates_by_day = dict.fromkeys(conditions_to_analyze, None)
for condition in conditions_to_analyze:
    ITI_lick_rates_all = np.zeros((num_animals_by_condition[condition],
                                  num_trials[condition])
                                  )
    ITI_lick_rates_all[:] = np.nan
    reward_bout_dur_all = np.zeros((num_animals_by_condition[condition],
                                  num_trials[condition])
                                  )
    reward_bout_dur_all[:] = np.nan
    prelearning_lick_rates[condition] = dict.fromkeys(animals_by_condition[condition], [])

    reward_bout_dur_before_after[condition] = {'beginning': [], 'end':[]}
    ITI_lick_rates_by_day[condition] = np.zeros((num_animals_by_condition[condition],
                                                   8)
                                                )
    ITI_lick_rates_by_day[condition][:] = np.nan
    for a_idx, animal in enumerate(animals_by_condition[condition]):
        animal_session_df = session_df[session_df['animal'] == animal].copy()
        ITI_lick_rates = []
        reward_bout_dur = []
        trial_count = 0
        excluded_trials_single_animal = []
        first_6_reward_bout_dur = []
        last_6_reward_bout_dur = []
        for ics, animal_session_df_row in animal_session_df.iterrows():
            cue_times = animal_session_df_row['CS_plus_on']
            reward_times = animal_session_df_row['reward_time']
            lick_on = animal_session_df_row['lick_on']
            lick_off = animal_session_df_row['lick_off']
            if len(lick_on) > 0:
                lick_bout_start, lick_bout_stop = lpf.boutStartStop(lick_on,
                                                                    lick_off,
                                                                    maxIBI = dc.bout_threshold
                                                                    )
                reward_lick_bout_end = [lick_bout_stop[lick_bout_stop>rew][0]
                                        if len(lick_bout_stop[lick_bout_stop>rew]) > 0
                                        else np.nan
                                        for rew
                                        in reward_times
                                        ]

                lick_spans_reward =  [1
                                      if ((lick_on[lick_off > rew].size> 0)
                                          and (lick_on[lick_off > rew][0] <rew)
                                          )
                                      else 0
                                      for rew
                                      in reward_times
                                      ]
                reward_lick_bout_start = [rew
                                          if lck_spn_rew
                                          else lick_on[lick_on> rew][0]
                                          if len(lick_on[lick_on> rew]) > 0
                                          else np.nan
                                          for (rew,
                                               lck_spn_rew
                                               )
                                          in zip(reward_times,
                                                 lick_spans_reward
                                                 )
                                          ]
                reward_bout_duration =  [bt_end - bt_strt

                                        for (
                                             bt_strt,
                                             bt_end
                                             )
                                        in zip (reward_lick_bout_start,
                                                reward_lick_bout_end
                                                )
                                        ]

                ITI_start = [0, *reward_lick_bout_end[:-1]]
                ITI_end = cue_times
                rate_by_day = []
                reward_bout_dur_daily = []
                for (idx, (beg, end, rew, rew_bout, rew_bt_dur)) in enumerate(zip(ITI_start,
                                                                      ITI_end,
                                                                      reward_times,
                                                                      reward_lick_bout_start,
                                                                      reward_bout_duration)
                                                                  ):
                    if ((rew_bout> (rew+5)) & (trial_count == 0)):
                        excluded_trials_single_animal.append(idx+1)
                    elif ((animal == '60s_F3') & (animal_session_df_row['day_num'] ==1) & (idx<= 2)):
                        #animal is grabbing spout causing artifiactial "long" bout durations during the first trials, so exclude these trials
                        ITI_lick_rates.append(np.nan)
                        reward_bout_dur.append(np.nan)
                        reward_bout_dur_daily.append(np.nan)
                    else:
                        ITI_len = end - beg
                        trial_count+=1
                        lick_rate = (len(lick_on[((lick_on> beg) & (lick_on < end))])/ITI_len)*1
                        ITI_lick_rates.append(lick_rate)
                        rate_by_day.append(lick_rate)
                        reward_bout_dur.append(rew_bt_dur)
                        reward_bout_dur_daily.append(rew_bt_dur)
                        if animal in learned_trial_data_lick[condition].keys():
                            if trial_count == learned_trial_data_lick[condition][animal]:
                                prelearning_lick_rates[condition][animal] = np.nanmedian(ITI_lick_rates)

                        if ((idx == (first_last_per_day - 1)) or (condition == '3600s' and idx == 1)):
                            first_6_reward_bout_dur.append(np.nanmean(reward_bout_dur_daily))
                last_6_reward_bout_dur.append(np.nanmean(reward_bout_dur_daily[-first_last_per_day:]))
                if (animal_session_df_row['day_num'] <=8) :
                    ITI_lick_rates_by_day[condition][a_idx, animal_session_df_row['day_num'] - 1] = np.nanmean(rate_by_day)
            else:
                for idx, rew in enumerate(reward_times):
                    ITI_lick_rates.append(np.nan)
                    reward_bout_dur.append(np.nan)
                print(animal_session_df_row['animal'])
                print(animal_session_df_row['day_num'])
        if len(first_6_reward_bout_dur) == 0:
            print(f'{animal} no first 6 rewards')
        reward_bout_dur_before_after[condition]['beginning'].append(np.nanmean(first_6_reward_bout_dur))
        reward_bout_dur_before_after[condition]['end'].append(np.nanmean(last_6_reward_bout_dur))
        if len(excluded_trials_single_animal) > 0:
            excluded_trials[animal] = excluded_trials_single_animal
        if len(ITI_lick_rates) < num_trials[condition]:
            num_to_pad =  num_trials[condition] - len(ITI_lick_rates)
            ITI_lick_rates =np.pad([float(x) for x in ITI_lick_rates], [(0, num_to_pad)], mode= 'constant', constant_values = np.nan)
            reward_bout_dur =np.pad([float(x) for x in reward_bout_dur], [(0, num_to_pad)], mode= 'constant', constant_values = np.nan)
        elif len(ITI_lick_rates) > num_trials[condition]:
            ITI_lick_rates =ITI_lick_rates[:num_trials[condition]]
            reward_bout_dur = reward_bout_dur[:num_trials[condition]]
        else:
            ITI_lick_rates =ITI_lick_rates
            reward_bout_dur = reward_bout_dur
        ITI_lick_rates_all[a_idx, :] = ITI_lick_rates
        reward_bout_dur_all[a_idx, :] = reward_bout_dur
    ITI_lick_rates_dict[condition] = ITI_lick_rates_all
    reward_bout_dur_dict[condition] = reward_bout_dur_all
#%%
"""
EXTENDED DATA FIGURE 8:
 - PANEL F (RIGHT SIDE): REWARD CONSUMPTION BOUT DURATION 30S AND 60S FIRST 6 TRIALS LAST 6 TRIALS/SESSION
 (bar graphs for right side of this panel generated in 'ExtendedDataFig8fk.py' script)
"""

for condition in ['60s', '30s']:

    figax_before_after_lick = ff.plot_paired_bars_from_dicts_or_list(reward_bout_dur_before_after[condition],
                                                                   condition_colors = {'beginning': 'black', 'end': dc.colors_for_conditions[condition]},
                                                                   ylabel =f'reward bout duration beginning and end of session {condition}',
                                                                   data_is_nested_dict = False,
                                                                   data_is_regular_dict = True,
                                                                   labels = ['beginning', 'end'],
                                                                   axsize = dc.axsize_bars_2,
                                                                   save_stats = save_stats,
                                                                   fig_path = fig_path_ExtDataFig8fk,
                                                                   save_fig = save_figs,
                                                                   )
#%%
"""
EXTENDED DATA FIGURE 8:
 - PANEL G: ITI LICK RATE ACROSS DAYS 30, 60, 300, 600, 3600S
"""

fig_lickrates_by_day, ax_lickrates_by_day = plt.subplots()
for condition, lckrates in ITI_lick_rates_by_day.items():
    mean_licks_single_con = np.nanmean(lckrates, axis = 0)
    sem_licks_single_con = stats.sem(lckrates, axis = 0, ddof = 1, nan_policy = 'omit')
    xaxis = np.arange(8)+1
    ax_lickrates_by_day.errorbar(xaxis,
                     mean_licks_single_con,
                     yerr = sem_licks_single_con,
                     color = dc.colors_for_conditions[condition],
                     marker = 'o',
                     label = f'{condition}',
                     markersize = 5,
                     markeredgewidth = 0,
                     linewidth = 0.5,
                     elinewidth = 0.25
                     )
ax_lickrates_by_day.set_ylabel('ITI lick rate (Hz)')
ax_lickrates_by_day.set_xlabel('day')
ax_lickrates_by_day.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ff.standardize_plot_graphics(ax_lickrates_by_day)
ff.set_ax_size_inches(dc.axsize_timecourse[0], dc.axsize_timecourse[1],  ax_lickrates_by_day)
if save_figs:
    fig_lickrates_by_day.savefig(os.path.join(fig_path_ExtDataFig8fk,
                                              f'ITI_lickrates across days_{conditions_to_analyze}.pdf'),
                                 transparent = True,
                                 bbox_inches = 'tight',
                                 )
#%%
"""
EXTENDED DATA FIGURE 8:
 - PANEL H: ITI LICK RATE PRELEARNING 30, 60, 300, 600, 3600S
 - PANEL I: ITI LICK RATE PRELEARNING vs IRI WITH FITLINE LOG PLOT 30, 60, 300, 600, 3600S
"""
condition_IRIs = {'30s': 34.25,
              '60s': 64.25,
                '300s': 304.25,
                '600s': 604.25,
                '3600s': 3604.25,
                }
axsize = dc.axsize_timecourse
markersize_means = 5

prelearning_lick_rates_regular_dict = {iti: [y
                                             for (x, y)
                                             in lck_prelrn.items()
                                             if ((not isinstance(y, list)) and (x != '600sD_F7'))
                                             ]
                                         for (iti,
                                              lck_prelrn)
                                         in prelearning_lick_rates.items()
                                        }

figax_learned_trial = ff.plotBarsFromDict(prelearning_lick_rates_regular_dict,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='ITI lick rate pre-learning',
                                           plot_stats = False,
                                           save_stats = save_stats,
                                           data_is_nested_dict = False,
                                           data_is_regular_dict = True,
                                           save_fig = save_figs,
                                           axsize = dc.axsize_bars_5,
                                           fig_path = fig_path_ExtDataFig8fk,
                                           )


# IRI vs prelearn lick rates means
fig_prelearnvsIRI_means_only, ax_prelearnvsIRI_means_only = plt.subplots()

prelearn_lickrate_means_list = []
IRIs_list = []
for condition, prelrn_lick in prelearning_lick_rates_regular_dict.items():
    prelearn_lickrate_mean = np.mean(prelrn_lick)
    prelearn_lickrate_error = np.std(prelrn_lick, ddof = 1)
    ax_prelearnvsIRI_means_only.errorbar(condition_IRIs[condition],
                                             prelearn_lickrate_mean,
                                             marker = 'o',
                                             linewidth = 0,
                                             color = dc.colors_for_conditions[condition],
                                             markersize = markersize_means,
                                             yerr = prelearn_lickrate_error,
                                             ecolor = dc.colors_for_conditions[condition],
                                             elinewidth = 0.5
                                             )
    prelearn_lickrate_means_list.append(prelearn_lickrate_mean)
    IRIs_list.append(condition_IRIs[condition])

IRIs_log = [np.log10(x)
            for x
            in IRIs_list
            ]
prelearn_lickrates_log = [np.log10(y)
            for y
            in prelearn_lickrate_means_list
            ]
fit_line_prelearnvsIRI_mean = stats.linregress(IRIs_log, prelearn_lickrates_log)
ax_prelearnvsIRI_means_only.set_xscale("log", base=10)
ax_prelearnvsIRI_means_only.set_yscale("log", base=10)
axlim_IRIs = np.arange(ax_prelearnvsIRI_means_only.get_xlim()[0], ax_prelearnvsIRI_means_only.get_xlim()[1], 1)
ax_prelearnvsIRI_means_only.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_prelearnvsIRI_means_only.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_prelearnvsIRI_means_only.plot(axlim_IRIs,
                                np.power(10, fit_line_prelearnvsIRI_mean.intercept)*np.power(axlim_IRIs, fit_line_prelearnvsIRI_mean.slope),
                                color = 'k',
                                linewidth = 1,
                                linestyle = 'solid',
                                alpha = 1,
                                )
ax_prelearnvsIRI_means_only.set_ylim([.01, None])
ax_prelearnvsIRI_means_only.set_xlim([10, None])
log_fit_title= 'log_fit'


ax_prelearnvsIRI_means_only.set_title(f' all animals r = {fit_line_prelearnvsIRI_mean.rvalue} p = {fit_line_prelearnvsIRI_mean.pvalue}')
ax_prelearnvsIRI_means_only.set_ylabel('lick rate in ITI (Hz)\n(prelearning)')
ax_prelearnvsIRI_means_only.set_xlabel('IRI')

ff.standardize_plot_graphics(ax_prelearnvsIRI_means_only)
ff.set_ax_size_inches(dc.axsize_timecourse[0], dc.axsize_timecourse[1],  ax_prelearnvsIRI_means_only)
if save_figs:
    fig_prelearnvsIRI_means_only.savefig(os.path.join(fig_path_ExtDataFig8fk,
                                                      f'prelearn_lickrates_vs_IRI_{conditions_to_analyze}_prelearning_meansonly_{log_fit_title}.pdf'),
                                        transparent = True,
                                        bbox_inches = 'tight',
                                        )

#%%
"""
EXTENDED DATA FIGURE 8:
 - PANEL J: TRIALS TO LEARN VS PRELEARNING ITI LICK RATE ALL CONDITIONS
 - PANEL K: TRIALS TO LEARN VS PRELEARNING ITI LICK RATE ALL CONDITIONS 60 300 600
"""

fig_prelearnvsTTL_means_only, ax_prelearnvsTTL_means_only = plt.subplots()

prelearn_lickrate_means_list = []
trials_to_learn_means_list = []
prelearn_lickrates_list_all = []
trials_to_learn_list_all = []
prelearn_lickrates_list_all = []
trials_to_learn_list_all = []
markersize = 3
color_by_sex = False
for condition, animal_ttls in learned_trial_data_lick.items():
    fig_prelearnvsTTL_single_con, ax_prelearnvsTTL_single_con = plt.subplots()
    prelearn_lickrates_list_single_con = []
    trials_to_learn_list_single_con = []
    for animal, trlTL in animal_ttls.items():
        if animal != '600sD_F7':
            ax_prelearnvsTTL_single_con.plot(prelearning_lick_rates[condition][animal],
                     trlTL,
                     marker = 'o',
                     linewidth = 0,
                     color = dc.colors_for_conditions[condition],
                     markersize = 3)
            prelearn_lickrates_list_single_con.append(prelearning_lick_rates[condition][animal])
            trials_to_learn_list_single_con.append(trlTL)
            prelearn_lickrates_list_all.append(prelearning_lick_rates[condition][animal])
            trials_to_learn_list_all.append(trlTL)
    mean_TLL_single_con = np.nanmean(trials_to_learn_list_single_con)
    std_TLL_single_con = np.std(trials_to_learn_list_single_con, ddof = 1)
    mean_prelearn_lickrates_single_con = np.nanmean(prelearn_lickrates_list_single_con)
    std_prelearn_lickrates_single_con = np.std(prelearn_lickrates_list_single_con, ddof = 1)
    prelearn_lickrate_means_list.append(mean_prelearn_lickrates_single_con)
    trials_to_learn_means_list.append(mean_TLL_single_con)
    ax_prelearnvsTTL_means_only.errorbar(mean_prelearn_lickrates_single_con,
                                             mean_TLL_single_con,
                                             marker = 'o',
                                             linewidth = 0,
                                             color = dc.colors_for_conditions[condition],
                                             markersize = markersize_means,
                                             yerr = std_TLL_single_con,
                                             xerr = std_prelearn_lickrates_single_con,
                                             ecolor = dc.colors_for_conditions[condition],
                                             elinewidth = 0.5
                                             )

    log_fit_title= 'log_fit'
    prelearn_lickrates_log_single_con = [np.log10(x)
                for x
                in prelearn_lickrates_list_single_con
                ]
    ttl_log_single_con = [np.log10(y)
                for y
                in trials_to_learn_list_single_con
                ]
    fit_line_prelearnvsTTL_single_con = stats.linregress(prelearn_lickrates_log_single_con, ttl_log_single_con)
    print(f'{condition} slope: {fit_line_prelearnvsTTL_single_con.slope}')
    print(f'{condition} intercept: {fit_line_prelearnvsTTL_single_con.intercept}')
    print(f'{condition} pval: {fit_line_prelearnvsTTL_single_con.pvalue}')
    print(f'{condition} r: {fit_line_prelearnvsTTL_single_con.rvalue}')
    print(f'{condition} r^2 =: {fit_line_prelearnvsTTL_single_con.rvalue**2}')
    print(f'{condition} n =: {len(prelearn_lickrates_list_single_con)}')

    ax_prelearnvsTTL_single_con.set_xscale("log", base=10)
    ax_prelearnvsTTL_single_con.set_yscale("log", base=10)

    axlim_single_con = np.arange(ax_prelearnvsTTL_single_con.get_xlim()[0], ax_prelearnvsTTL_single_con.get_xlim()[1], 0.05)
    ax_prelearnvsTTL_single_con.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_prelearnvsTTL_single_con.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_prelearnvsTTL_single_con.plot(axlim_single_con,
                                    np.power(10, fit_line_prelearnvsTTL_single_con.intercept)*np.power(axlim_single_con, fit_line_prelearnvsTTL_single_con.slope),
                                    color = 'k',
                                    linewidth = 1,
                                    linestyle = 'solid',
                                    alpha = 1,
                                    )
    ax_prelearnvsTTL_single_con.set_xlim([0.01, None])
    ax_prelearnvsTTL_single_con.set_title(f' {condition} r = {fit_line_prelearnvsTTL_single_con.rvalue} p = {fit_line_prelearnvsTTL_single_con.pvalue}')
    ax_prelearnvsTTL_single_con.set_ylabel('learned trial')
    ax_prelearnvsTTL_single_con.set_xlabel('ITI lick rate (prelearning)')
    ff.standardize_plot_graphics(ax_prelearnvsTTL_single_con)
    ff.set_ax_size_inches(dc.axsize_timecourse[0], dc.axsize_timecourse[1],  ax_prelearnvsTTL_single_con)
    if save_figs:
        fig_prelearnvsTTL_single_con.savefig(os.path.join(fig_path_ExtDataFig8fk, f'prelearn_lickrates_vs_trials_to_learn single condition {condition}_{log_fit_title}.pdf'),
                        transparent = True,
                        bbox_inches = 'tight')


prelearn_lickrates_log_means = [np.log10(x)
            for x
            in prelearn_lickrate_means_list
            ]
ttl_log_means = [np.log10(y)
            for y
            in trials_to_learn_means_list
            ]

fit_line_prelearnvsTTL_mean = stats.linregress(prelearn_lickrates_log_means, ttl_log_means)
ax_prelearnvsTTL_means_only.set_xscale("log", base=10)
ax_prelearnvsTTL_means_only.set_yscale("log", base=10)
axlim_prelearnTTL = np.arange(ax_prelearnvsTTL_means_only.get_xlim()[0], ax_prelearnvsTTL_means_only.get_xlim()[1], 0.05)
ax_prelearnvsTTL_means_only.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_prelearnvsTTL_means_only.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_prelearnvsTTL_means_only.plot(axlim_prelearnTTL,
                                np.power(10, fit_line_prelearnvsTTL_mean.intercept)*np.power(axlim_prelearnTTL, fit_line_prelearnvsTTL_mean.slope),
                                color = 'k',
                                linewidth = 1,
                                linestyle = 'solid',
                                alpha = 1,
                                )

ax_prelearnvsTTL_means_only.set_title(f' all-means only r = {fit_line_prelearnvsTTL_mean.rvalue} p = {fit_line_prelearnvsTTL_mean.pvalue}')
ax_prelearnvsTTL_means_only.set_ylabel('learned trial')
ax_prelearnvsTTL_means_only.set_xlabel('ITI lick rate (Hz)\n (pre-learning)')
ff.standardize_plot_graphics(ax_prelearnvsTTL_means_only)
ff.set_ax_size_inches(dc.axsize_timecourse[0], dc.axsize_timecourse[1],  ax_prelearnvsTTL_means_only)
if save_figs:
    fig_prelearnvsTTL_means_only.savefig(os.path.join(fig_path_ExtDataFig8fk, f'prelearnlickrates-means-_vs_trials_to_learn{conditions_to_analyze}_log_fit.pdf'),
                    transparent = True,
                    bbox_inches = 'tight')