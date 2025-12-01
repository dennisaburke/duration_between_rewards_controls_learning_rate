# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

FIGURE 7: Partial reinforcement scales learning rate by increasing the inter-reward interval.
"""


"""
imports
"""
import os
import pingouin
import scipy.stats as stats
import numpy as np

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
fig_path_Figure7_ExtDataFig10 = os.path.join(figure_path_root, r'Figure7_ExtDataFig10')

save_figs = True
save_stats = True

#%%
"""
load and prepare data

"""
nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,
                                                                  [ '30s',
                                                                    '60s',
                                                                    '300s',
                                                                    '600s',
                                                                    '60s-50%',
                                                                    '60s-10%',
                                                                    ],
                                                                  )
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )
#%%
df_behavior_days_CSplus = lp.get_behavior_days_CSplus_df(df)
df_behavior_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
nonlearners_list = lpf.get_nonlearners(df_behavior_days_CSplus)
df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)
df_dlight_trials_CSplus = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_df(df))
df_dlight_trials_CSplus_learners = lp.subset_dopamine_animals(df_behavior_trials_CSplus_learners)
df_dlight_trials_CSplus_nonlearners = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_nonlearners_df(df))
df_rewards = df_behavior_trials_CSplus[df_behavior_trials_CSplus['trial_type'] == 'reward'].copy()
df_rewards = lpf.cumLickTrialCount(df_rewards, grouping_var = ['animal', 'cue_type', 'trial_type'])
df_behavior_trials_CSplus_10percentexcl = df_behavior_trials_CSplus_learners[~df_behavior_trials_CSplus_learners['animal'].isin(['60s-10%D_M2'])] #outlier dopamine 60s-10% see extdata10j
df_dlight_trials_CSplus_10percentexcl = lp.subset_dopamine_animals(df_behavior_trials_CSplus_10percentexcl)


#%%
"""
FIGURE 7
 - PANEL C: REWARDS TO LEARN 60S-50% VS 60S
 - PANEL B: 60S-50% INDIVIDUAL ANIMAL LICK AND DA CUMSUM AS FUNCTION OF REWARDED TRIALS
"""
#PANEL C
conditions_to_plot = [ '60s',
                      '60s-50%',
                      ]
learned_trial_data_learners = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus,
                                                      conditions_to_plot=conditions_to_plot,
                                                      get_DA_learned_trial = True,
                                                      nonlearners_list = nonlearners_list,
                                                      )
learned_trial_lick_learners = learned_trial_data_learners['learned_trial_lick']
rewards_to_learn_dict_learners = lpf.calculate_rewards_to_learn_from_learned_trials(learned_trial_lick_learners,
                                                                                    df_behavior_trials_CSplus_learners,
                                                                                    conditions_to_plot,
                                                                                    )
figax_learned_trial = ff.plotBarsFromDict(rewards_to_learn_dict_learners,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='rewards to learn',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure7_ExtDataFig10,
                                           )
#PANEL B
conditions_to_plot = ['60s-50%',]
learned_trial_data_rewardsonly = ff.getCumSumLearnedTrialsAndPlot(df_rewards,
                                                      conditions_to_plot=conditions_to_plot,
                                                      colors_for_conditions = dc.colors_for_conditions,
                                                      colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                      plot_all_individuals = True,
                                                      get_DA_learned_trial = True,
                                                      use_trial_normalized_y = True,
                                                      linewidth_learned_trial = 0,
                                                      linewidth_DA_trial = 0,
                                                      save_fig = False,
                                                      fig_path = fig_path_Figure7_ExtDataFig10,
                                                      plot_on_2_lines = False,
                                                      sharey_DA = True,
                                                      ylim_lick = [-0.275, 5.5],
                                                      ylim_DA = [-0.01375, 0.275],
                                                      sharex = True,
                                                      axsize = dc.axsize_cumsum_all_individuals_DA,
                                                      )


learned_trial_data_all_DA = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus,
                                                                  conditions_to_plot=conditions_to_plot,
                                                                  get_DA_learned_trial = True,
                                                                  nonlearners_list = nonlearners_list,
                                                                  )
rewards_to_learn_DA_dict_all = lpf.calculate_rewards_to_learn_from_learned_trials(learned_trial_data_all_DA['learned_trial_DA'],
                                                                                       df_dlight_trials_CSplus,
                                                                                       conditions_to_plot,
                                                                                       )
#plot rewards to learn DA and behavior calculated from trials to learn above on individual reward trial cumsums
reward_ax = learned_trial_data_rewardsonly['ax']
for ax_indiv in reward_ax.reshape(-1):
    if ax_indiv.get_title() in rewards_to_learn_dict_learners['60s-50%'].keys():
        if not isinstance(rewards_to_learn_dict_learners['60s-50%'][ax_indiv.get_title()], list):
            ax_indiv.axvline(rewards_to_learn_dict_learners['60s-50%'][ax_indiv.get_title()],
                             color = 'black',
                             linestyle = 'solid',
                             linewidth = 0.35,
                             )
    if ax_indiv.get_title() in rewards_to_learn_DA_dict_all['60s-50%'].keys():
        ax_indiv.axvline(rewards_to_learn_DA_dict_all['60s-50%'][ax_indiv.get_title()],
                         color = 'black',
                         linestyle = 'dashed',
                         linewidth = 0.35,
                         )
fig = ax_indiv.get_figure()
if save_figs:
    fig.savefig(os.path.join(fig_path_Figure7_ExtDataFig10, f'rewards_to_learn_individualcumsums {conditions_to_plot}.pdf'),
                    transparent = True,
                    bbox_inches = 'tight',
                    )

#%%
"""
FIGURE 7
 - PANEL D: LICK RATE ACROSS TRIALS LEARNERS ONLY + ASYMPTOTE BAR GRAPHS
 - PANEL E: CUE DA ACROSS TRIALS LEARNERS ONLY + ASYMPTOTE BAR GRAPHS
"""
conditions_to_plot = ['60s',
                      '60s-50%',
                      ]

#PANEL D LEFT
plot_lick = True
fig_lick_avg_trials_both2, ax_lick_avg_trials_both2 = ff.plotDALickOverTime(df_behavior_trials_CSplus_learners,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = 'learners',
                                                                          fig_path  = fig_path_Figure7_ExtDataFig10,
                                                                          ylim_lick = [None, 6],
                                                                          xlim_lick = [-5, None],
                                                                          )
#PANEL D RIGHT
range_to_plot = {'60s': [301, 400],
                 '60s-50%': [501, 600],
                 }
plot_lick_or_DA = 'lick'
fig_cueDA_means, ax_cueDA_means = ff.compare_asymptote_bars(df_behavior_trials_CSplus_learners,
                                                            conditions_to_plot = conditions_to_plot,
                                                            range_to_plot = range_to_plot,
                                                            condition_colors = dc.colors_for_conditions,
                                                            plot_lick_or_DA = plot_lick_or_DA,
                                                            save_stats = save_stats,
                                                            ylabel ='lick rate to cue (Hz)\nlast 100 trials',
                                                            axsize = dc.axsize_bars_2,
                                                            save_fig = save_figs,
                                                            fig_path = fig_path_Figure7_ExtDataFig10,
                                                            )
#PANEL E RIGHT
plot_lick = False
plot_DA_cue = True
linewidth_DA_cue = 0.25
fig_lick_avg_trials_both, ax_lick_avg_trials_both = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          plot_DA_cue= plot_DA_cue,
                                                                          linewidth_DA_cue = linewidth_DA_cue,
                                                                          colors_for_conditions_DA = dc.colors_for_conditions,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = 'learners',
                                                                          ylim_DA = [-0.1, 0.5],
                                                                          xlim_DA = [-5, None],
                                                                          fig_path  = fig_path_Figure7_ExtDataFig10,
                                                                          )
#PANEL E LEFT
range_to_plot = {'60s': [301, 400],
                 '60s-50%': [501, 600],
                 }
plot_lick_or_DA = 'DA'
fig_cueDA_means, ax_cueDA_means = ff.compare_asymptote_bars(df_dlight_trials_CSplus_learners,
                                                            conditions_to_plot = conditions_to_plot,
                                                            range_to_plot = range_to_plot,
                                                            condition_colors = dc.colors_for_conditions,
                                                            plot_lick_or_DA = plot_lick_or_DA,
                                                            save_stats = save_stats,
                                                            ylabel ='cue DA\nlast 100 trials',
                                                            axsize = dc.axsize_bars_2,
                                                            save_fig = save_figs,
                                                            fig_path = fig_path_Figure7_ExtDataFig10,
                                                            )
#%%
"""
FIGURE 7
 - PANEL F: LEARNERS AND NONLEARNERS CUMSUM LICK AND DA
"""
plot_lick = True
plot_DA_cue = True
plot_cumsum = True
linewidth_lick = 1.5
linewidth_DA_cue = 1.5
linestyle_DA_cue = (0, (4.5, 1.5))
renamed_conditions_omissions_df = lp.get_renamed_learners_nonlearners_df(df_dlight_trials_CSplus_learners,
                                                                      df_dlight_trials_CSplus_nonlearners,
                                                                      condition_to_replace = '60s-50%',
                                                                      )
colors_lick = {'learners': '#ad2372',
               'non-learners': 'black'}
colors_da = {'learners': '#ad2372',
               'non-learners': 'black'}
conditions_to_plot =['learners',
                     'non-learners',
                     ]
fig_lick_avg_trials_both2, ax_lick_avg_trials_both2 = ff.plotDALickOverTime(renamed_conditions_omissions_df,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions =colors_lick,
                                                                          plot_cumsum = plot_cumsum,
                                                                          plot_lick = plot_lick,
                                                                          linewidth_DA_cue =linewidth_DA_cue,
                                                                          linestyle_DA_cue =linestyle_DA_cue,
                                                                          plot_DA_cue = plot_DA_cue,
                                                                          linewidth_lick = linewidth_lick,
                                                                          colors_for_conditions_DA = colors_da,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = '',
                                                                          fig_path  = fig_path_Figure7_ExtDataFig10,
                                                                          )
#%%
"""
EXTENDED DATA FIGURE 10
 - PANEL H: 60S-10% INDIVIDUAL CUMSUM WITH DA AND BEHAVIOR "LEARNED TRIAL". NO NONLEARNER EXCLUSIONS
FIGURE 7
 - PANEL I: 60S-10% CUMSUM OF DOPAMINE ALIGNED TO DOPAMINE LEARNED TRIAL
"""


learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_10percentexcl,
                                                      conditions_to_plot='60s-10%',
                                                      colors_for_conditions = dc.colors_for_conditions,
                                                      colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                      plot_all_individuals = True,
                                                      get_DA_learned_trial = True,
                                                      use_trial_normalized_y = True,
                                                      save_fig = save_figs,
                                                      fig_path = fig_path_Figure7_ExtDataFig10,
                                                      plot_on_2_lines = False,
                                                      sharey_DA = True,
                                                      ylim_lick =[-0.1, 1],
                                                      ylim_DA = [-0.01, 0.1],
                                                      sharex = False,
                                                      axsize = dc.axsize_cumsum_all_individuals_DA)

learned_trial_lick = learned_trial_data['learned_trial_lick']
learned_trial_DA = learned_trial_data['learned_trial_DA']

plot_cumsum = True
plot_lick = False
plot_DA_cue = True
align_to_DA = True
align_to_lick = False
peak_or_auc = 'auc'
norm_to_max_individual_rewards = 0
norm_to_one_individual = True
ylim_left = [None, 0.5]
aligned_lickDAtimecourses_60s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_10percentexcl,
                                                                conditions_to_plot = ['60s-10%'],
                                                                learned_trials = learned_trial_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                peak_or_auc = peak_or_auc,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                align_to_DA = align_to_DA,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = ylim_left,
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure7_ExtDataFig10,
                                                                )
#%%
"""
FIGURE 7
 - PANEL H: EXAMPLE 60S-10% ANIMAL BEFORE AND AFTER DA LEARNED TRIAL
"""
beforeafterDA_fig, beforeafterDA_ax = ff.plotPSTHbyBeforeAfter(df_dlight_trials_CSplus_10percentexcl,
                                                                dc.cumsum_examples_DA['60s-10%'],
                                                                learned_trial = learned_trial_lick['60s-10%'][dc.cumsum_examples_DA['60s-10%']],
                                                                trial_before_after = (10,10),
                                                                plot_lick_raster = True,
                                                                plot_lick_PSTH = True,
                                                                plot_DA_heatmap = True,
                                                                plot_DA_PSTH = True,
                                                                ylim_DA = [None, 7.5],
                                                                color_DA_PSTH = dc.colors_for_conditions['60s-10%'],
                                                                color_lick_PSTH = dc.colors_for_conditions['60s-10%'],
                                                                fig_path = fig_path_Figure7_ExtDataFig10,
                                                                save_fig = save_figs,
                                                                )
#%%
"""
FIGURE 7
 - PANEL J: REWARDS TO LEARN DA 60S-10%, 60S, 600S
"""
conditions_to_plot = ['60s-10%',
                       '60s',
                      '600s',
                      ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_10percentexcl,
                                                      conditions_to_plot=conditions_to_plot,
                                                      plot_all_individuals = False,
                                                       colors_for_conditions= dc.colors_for_conditions,
                                                       colors_for_conditions_DA = dc.colors_for_conditions,
                                                      get_DA_learned_trial = True,
                                                      use_trial_normalized_y = True,
                                                      )
learned_trial_DA = learned_trial_data['learned_trial_DA']
rewards_to_learn_DA_dict = lpf.calculate_rewards_to_learn_from_learned_trials(learned_trial_DA,
                                                                              df_dlight_trials_CSplus_10percentexcl,
                                                                              conditions_to_plot,
                                                                              )
figax_learned_trial = ff.plotBoxplotFromDict(rewards_to_learn_DA_dict,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='rewards to learn DA',
                                           plot_median_and_IQR = True,
                                           save_stats = save_figs,
                                           axsize = dc.axsize_bars_3,
                                           save_fig = save_figs,
                                           ylim = [-1, 61],
                                           fig_path = fig_path_Figure7_ExtDataFig10,
                                           )
#stats
rewards_to_learn_DA_lists  = {x: [y
                                   for y
                                   in rewards_to_learn_DA_dict[x].values()
                                   if type(y) is not list
                                   ]
                               for x
                               in rewards_to_learn_DA_dict
                               }


MannWhitneyU_60_vs_10percent_da  = pingouin.mwu(rewards_to_learn_DA_lists['60s-10%'], rewards_to_learn_DA_lists['60s'], alternative='two-sided')
MannWhitneyU_600_vs_10percent_da  = pingouin.mwu(rewards_to_learn_DA_lists['600s'], rewards_to_learn_DA_lists['60s-10%'], alternative='two-sided')


#%%
"""
EXTENDED DATA FIGURE 10
 - PANEL A: ACTUAL VS PREDICTED REWARDS TO LEARN 60S-50%
"""
#get fit line to get predicted trial to learn for IRI vs ICI
conditions_fitline = ['30s',
                        '60s',
                        '300s',
                        '600s',
                        ]
conditions_to_plot = ['30s',
                        '60s',
                        '300s',
                        '600s',
                        '60s-50%',
                        ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
rewards_to_learn_dict = lpf.calculate_rewards_to_learn_from_learned_trials(learned_trial_lick,
                                                                              df_behavior_trials_CSplus_learners,
                                                                              conditions_to_plot,
                                                                              )
ylim = [None, 320]
xlim = [10, 1001]
IRI_vs_trials_to_learn = ff.plot_IRI_vs_learned_trial_scatter(rewards_to_learn_dict,
                                      colors_for_conditions = dc.colors_for_conditions,
                                      conditions_to_plot = conditions_to_plot,
                                      conditions_for_fitline = conditions_fitline,
                                      xlim = xlim,
                                      ylim = ylim,
                                      ax_to_plot = None,
                                      axsize = dc.axsize_timecourse,
                                      save_fig = False,
                                      fig_path = fig_path_Figure7_ExtDataFig10,
                                      )
IRI_to_predict_IRI = 128.5
IRI_to_predict_ICI = 64.25

linear_Y_predict_IRI = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict_IRI, IRI_vs_trials_to_learn['fit_line'])
linear_Y_predict_ICI = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict_ICI, IRI_vs_trials_to_learn['fit_line'])

IRI_vs_trials_to_learn['ax'].axhline(linear_Y_predict_IRI,
                                    linestyle = (2,(4,4)),
                                    color = 'k',
                                    linewidth = 0.5,
                                    alpha = 1,
                                    )
IRI_vs_trials_to_learn['ax'].axhline(linear_Y_predict_ICI,
                                    linestyle = (2,(4,4)),
                                    color = 'k',
                                    linewidth = 0.5,
                                    alpha = 1,
                                    )
IRI_vs_trials_to_learn['ax'].set_ylabel('rewards to learn')
if save_figs:
    IRI_vs_trials_to_learn['fig'].savefig(os.path.join(fig_path_Figure7_ExtDataFig10,
                         'SixtySec50Percent-IRIvsLearnedTrialScatter_predicted_hlinesICIorIRI.pdf'),
                bbox_inches = 'tight',
                transparent = True,
                )

reward_to_learn60s50perc = [x for x in rewards_to_learn_dict['60s-50%'].values() if not isinstance(x, list)]
linear_Y_observed = np.mean(reward_to_learn60s50perc)
one_sample_ttest60s50perc_from_predicted_ICI = pingouin.ttest(reward_to_learn60s50perc, linear_Y_predict_ICI)
print(f'rewards to learn one-sample from predicted ICI: t = {one_sample_ttest60s50perc_from_predicted_ICI["T"].iloc[0]} p (uncorrected) = {one_sample_ttest60s50perc_from_predicted_ICI["p-val"].iloc[0]}')

one_sample_ttest60s50perc_from_predicted_IRI = pingouin.ttest(reward_to_learn60s50perc, linear_Y_predict_IRI)
print(f'rewards to learn DA one-sample from predicted IRI: t = {one_sample_ttest60s50perc_from_predicted_IRI["T"].iloc[0]} p (uncorrected) = {one_sample_ttest60s50perc_from_predicted_IRI["p-val"].iloc[0]}')

#%%
"""
EXTENDED DATA FIGURE 10
 - PANEL B: ACTUAL VS PREDICTED REWARDS TO LEARN DA 60S-50%
"""
conditions_to_plot = ['60s',
                      '600s',
                      '60s-50%',
                      ]

conditions_for_fitline = ['60s',
                          '600s',
                          ]
xlim = [10, 1000]
ylim = [1, 100]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      get_DA_learned_trial = True,
                                                      )
learned_trial_DA = learned_trial_data['learned_trial_DA']

rewards_to_learn_DA_dict = lpf.calculate_rewards_to_learn_from_learned_trials(learned_trial_DA,
                                                                              df_dlight_trials_CSplus_learners,
                                                                              conditions_to_plot,
                                                                              )

IRI_vs_trials_to_learn_DA = ff.plot_IRI_vs_learned_trial_scatter(rewards_to_learn_DA_dict,
                                      colors_for_conditions = dc.colors_for_conditions,
                                      conditions_to_plot = conditions_to_plot,
                                      conditions_for_fitline = conditions_for_fitline,
                                      xlim = xlim,
                                      ylim = ylim,
                                      ax_to_plot = None,
                                      axsize = dc.axsize_timecourse,
                                      save_fig = False,
                                      fig_path = fig_path_Figure7_ExtDataFig10
                                      )
IRI_to_predict_IRI = 128.5
IRI_to_predict_ICI = 64.25

linear_Y_predict_IRI_DA = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict_IRI, IRI_vs_trials_to_learn_DA['fit_line'])
linear_Y_predict_ICI_DA = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict_ICI, IRI_vs_trials_to_learn_DA['fit_line'])

IRI_vs_trials_to_learn_DA['ax'].axhline(linear_Y_predict_IRI_DA,
                                    linestyle = (2,(4,4)),
                                    color = 'k',
                                    linewidth = 0.5,
                                    alpha = 1,
                                    )
IRI_vs_trials_to_learn_DA['ax'].axhline(linear_Y_predict_ICI_DA,
                                    linestyle = (2,(4,4)),
                                    color = 'k',
                                    linewidth = 0.5,
                                    alpha = 1,
                                    )
IRI_vs_trials_to_learn_DA['ax'].set_ylabel('rewards to learn DA')
if save_figs:
    IRI_vs_trials_to_learn_DA['fig'].savefig(os.path.join(fig_path_Figure7_ExtDataFig10,
                         'SixtySec50Percent-IRIvsLearnedTrialScatter_predicted_hlinesICIorIRI_DA.pdf'),
                bbox_inches = 'tight',
                transparent = True,
               )
reward_to_learn60s50perc_DA = list(rewards_to_learn_DA_dict['60s-50%'].values())
linear_Y_observed_DA = np.mean(reward_to_learn60s50perc_DA)
one_sample_ttest60s50perc_from_predicted_ICI_DA = pingouin.ttest(reward_to_learn60s50perc_DA, linear_Y_predict_ICI_DA)
one_sample_ttest60s50perc_from_predicted_ICI_DA_scipy = stats.ttest_1samp(reward_to_learn60s50perc_DA, linear_Y_predict_ICI_DA)
print(f'rewards to learn DA one-sample from predicted ICI: t = {one_sample_ttest60s50perc_from_predicted_ICI_DA["T"].iloc[0]} p (uncorrected) = {one_sample_ttest60s50perc_from_predicted_ICI_DA["p-val"].iloc[0]}')
one_sample_ttest60s50perc_from_predicted_IRI_DA = pingouin.ttest(reward_to_learn60s50perc_DA, linear_Y_predict_IRI_DA)
print(f'rewards to learn DA one-sample from predicted IRI: t = {one_sample_ttest60s50perc_from_predicted_IRI_DA["T"].iloc[0]} p (uncorrected) = {one_sample_ttest60s50perc_from_predicted_IRI_DA["p-val"].iloc[0]}')
#%%
"""
EXTENDED DATA FIGURE 10
 - PANEL C: REWARDS TO LEARN DA
 - PANEL D: TRIALS TO LEARN 60S-50% VS 60S
 - PANEL E: TRIALS TO LEARN DA 60S-50% VS 60S
 - PANEL F: INDIVIDUAL ANIMAL CUMSUMS AS A FUNCTION OF TRIALS
"""
#PANEL C
conditions_to_plot = ['60s',
                      '60s-50%',
                      ]
learned_trial_data_learners_DA = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                                  conditions_to_plot=conditions_to_plot,
                                                                  get_DA_learned_trial = True,
                                                                  )
learned_trial_DA_learners = learned_trial_data_learners_DA['learned_trial_DA']
rewards_to_learn_DA_dict_learners = lpf.calculate_rewards_to_learn_from_learned_trials(learned_trial_DA_learners,
                                                                                       df_dlight_trials_CSplus_learners,
                                                                                       conditions_to_plot,
                                                                                       )
figax_learned_trial_DA = ff.plotBarsFromDict(rewards_to_learn_DA_dict_learners,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='DA learned reward (beh learners only)',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure7_ExtDataFig10,
                                           )
#PANEL D
learned_trial_data_learners = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                              conditions_to_plot=conditions_to_plot,
                                                              get_DA_learned_trial = True,
                                                              )
learned_trial_lick = learned_trial_data_learners['learned_trial_lick']
figax_learned_trial = ff.plotBarsFromDict(learned_trial_lick,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to learn',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure7_ExtDataFig10,
                                           )
#PANEL E
learned_trial_data_learners_dlight = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                              conditions_to_plot=conditions_to_plot,
                                                              get_DA_learned_trial = True,
                                                              )
figax_learned_trial = ff.plotBarsFromDict(learned_trial_data_learners_dlight['learned_trial_DA'],
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='DA learned trial',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure7_ExtDataFig10,
                                           )
#%%
"""
EXTENDED DATA FIGURE 10
 - PANEL G: LICK CUMSUM PLOTS 60S 10% 50% AND 100%
"""


plot_lick = True
plot_DA_cue = False
plot_cumsum = True
linewidth_lick = 1
fig_lick_avg_trials_both, ax_lick_avg_trials_both = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                          conditions_to_plot = ['60s','60s-50%','60s-10%'],
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_cumsum = plot_cumsum,
                                                                          plot_lick = plot_lick,
                                                                          plot_DA_cue = plot_DA_cue,
                                                                          linewidth_lick = linewidth_lick,
                                                                          axsize = (1.144, 1),
                                                                          save_figs = save_figs,
                                                                          fig_path  = fig_path_Figure7_ExtDataFig10,
                                                                          )

#%%
"""
EXTENDED DATA FIGURE 10
 - PANEL I: LICK RASTER +PSTH + DA FOR OUTLIER 60S-10%_F6 TO SHOW CUE DA ON DAY DESPITE LEARNED TRIAL ALGORITHM PICKING MUCH LATER TRIAL
"""
outlierDA_psth_fig, outlierDA_psth_ax =  ff.plotPSTHbyDay(df_behavior_trials_CSplus,
                                                        '60s-10%D_F6',
                                                        days_to_plot = [1,2],
                                                        plot_lick_raster = True,
                                                        plot_lick_PSTH = True,
                                                        plot_DA_heatmap = True,
                                                        plot_DA_PSTH = True,
                                                        color_DA_PSTH = dc.colors_for_conditions['60s-10%'],
                                                        color_lick_PSTH = dc.colors_for_conditions['60s-10%'],
                                                        colorplotmax = 2,
                                                        fig_path = fig_path_Figure7_ExtDataFig10,
                                                        save_fig = save_figs,
                                                        )
#%%
"""
EXTENDED DATA FIGURE 10
 - PANEL J: LICK RASTER +PSTH + DA FOR OUTLIER 60S-10%_M2 WITH NEGATIVE DOPAMINE DIPS TO CUE EXCLUDED FROM ANALYSIS
"""
outlierDA_psth_fig, outlierDA_psth_ax =  ff.plotPSTHbyDay(df_behavior_trials_CSplus,
                                                        '60s-10%D_M2',
                                                        days_to_plot = [31,32],
                                                        plot_lick_raster = True,
                                                        plot_lick_PSTH = True,
                                                        plot_DA_heatmap = True,
                                                        plot_DA_PSTH = True,
                                                        color_DA_PSTH = dc.colors_for_conditions['60s-10%'],
                                                        color_lick_PSTH = dc.colors_for_conditions['60s-10%'],
                                                        colorplotmax = 2,
                                                        fig_path = fig_path_Figure7_ExtDataFig10,
                                                        save_fig = save_figs,
                                                        )
excluded_animal_df = df_behavior_trials_CSplus[df_behavior_trials_CSplus['animal'] == '60s-10%D_M2']
outlierDA_cumsum_individual = ff.getCumSumLearnedTrialsAndPlot(excluded_animal_df,
                                                                conditions_to_plot = ['60s-10%'],
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                get_DA_learned_trial = True,
                                                                plot_all_individuals = True,
                                                                use_trial_normalized_y = True,
                                                                axsize = dc.axsize_cumsum_all_individuals_DA,
                                                                ylim_lick = [-2.5,2.5],
                                                                ylim_DA = [-0.025, 0.025],
                                                                sharey_DA = True,
                                                                lick_left = True,
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure7_ExtDataFig10,
                                                                 )
#%%
"""
EXTENDED DATA FIGURE 10:
 - PANEL K: REWARDS TO LEARN 60S-10%, 60S, 600S
"""
conditions_to_plot = ['60s-10%',
                       '60s',
                      '600s',
                      ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_10percentexcl,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
rewards_to_learn_dict = lpf.calculate_rewards_to_learn_from_learned_trials(learned_trial_lick,
                                                                              df_behavior_trials_CSplus_10percentexcl,
                                                                              conditions_to_plot,
                                                                              )
figax_learned_trial = ff.plotBoxplotFromDict(rewards_to_learn_dict,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='rewards to learn',
                                           plot_median_and_IQR = True,
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_3,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure7_ExtDataFig10,
                                           )
rewards_to_learn_lists  = {x: [y
                                for y
                                in rewards_to_learn_dict[x].values()
                                if type(y) is not list
                                ]
                            for x
                            in rewards_to_learn_dict
                            }


MannWhitneyU_60_vs_10percent  = pingouin.mwu(rewards_to_learn_lists['60s-10%'], rewards_to_learn_lists['60s'], alternative='two-sided')
MannWhitneyU_600_vs_10percent  = pingouin.mwu(rewards_to_learn_lists['600s'], rewards_to_learn_lists['60s-10%'], alternative='two-sided')

"""
EXTENDED DATA FIGURE 10 PANEL L SCRIPT IN SIMULATIONS FOLDERS
"""