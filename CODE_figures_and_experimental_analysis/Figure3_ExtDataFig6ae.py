# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

FIGURE 3: Learning rate scales proportionally with reward frequency across a range of trial spacing intervals.

"""

"""
imports
"""
import os

import functions.load_preprocess  as lp
import functions.default_configs as dc
import functions.figure_functions as ff
import functions.lick_photo_functions as lpf
import functions.stats_functions as sf

#%%
"""
set paths to data and outputs
"""
#set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

nwb_dir_path = r'..\DATA_experimental\001632'
figure_path_root = r'..\FIGURES'
fig_path_Figure3_ExtDataFig6ae = os.path.join(figure_path_root, r'Figure3_ExtDataFig6ae')

save_figs = True
save_stats = True

#%%
"""
load and prepare data
"""

nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,  ['30s', '60s', '300s', '600s'])
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )
#%%
#df_dlight_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
df_behavior_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)
nonlearners_list = lpf.get_nonlearners(lp.get_behavior_days_CSplus_df(df))
df_dlight_trials_CSplus_learners = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_learners_df(df))
df_first_80_trials = df_behavior_trials_CSplus[df_behavior_trials_CSplus['cue_trial_num'] <=80]

#%%
"""
FIGURE 3:
 - PANEL B: EXAMPLE 30 AND 300S LICK RASTERPLOT AND PSTH
"""
plot_lick = True
plot_DA = False

fig_ex_behavior_300s, ax_ex_behavior_300s = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_behavior['300s'],
                                                            color_lick_PSTH = dc.colors_for_conditions['300s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            axsize = dc.axsize_raster_PSTH_panel,
                                                            fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                            save_fig = save_figs,
                                                            )
fig_ex_behavior_30s, ax_ex_behavior_30s = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_behavior['30s'],
                                                            color_lick_PSTH = dc.colors_for_conditions['30s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            axsize = dc.axsize_raster_PSTH_panel,
                                                            fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                            save_fig = save_figs,
                                                            )
#%%
"""
FIGURE 3:
- PANEL C: EXAMPLE EXAMPLE 30 AND 300s CUMSUMS
"""
example_cumsum_lick_300s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus,
                                                                animal = dc.cumsum_examples_behavior['300s'],
                                                                color_lick = dc.colors_for_conditions['300s'],
                                                                axsize = dc.axsize_cumsum_examples,
                                                                use_trial_normalized_y = False,
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                                )
example_cumsum_lick_30s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus,
                                                                animal = dc.cumsum_examples_behavior['30s'],
                                                                color_lick = dc.colors_for_conditions['30s'],
                                                                axsize = dc.axsize_cumsum_examples,
                                                                use_trial_normalized_y = False,
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                                )
#%%
"""
FIGURE 3:
 - PANEL D: TIMECOURSE LICK BEHAVIOR 30 300s
"""
plot_lick = True
plot_cumsum = False
linewidth_lick = 0.25
ylim_lick = [None, 6]
xlim_lick = [-5, None]
fig_lick_avg_trials_both, ax_lick_avg_trials_both = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                          conditions_to_plot = ['300s',
                                                                                                '30s'],
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_cumsum = plot_cumsum,
                                                                          plot_lick = plot_lick,
                                                                          linewidth_lick = linewidth_lick,
                                                                          ylim_lick = ylim_lick,
                                                                          xlim_lick =xlim_lick,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          fig_path  = fig_path_Figure3_ExtDataFig6ae,
                                                                          )



#%%
"""
FIGURE 3:
 - PANEL E: TRIALS TO LEARN BAR GRAPH 30 300s
"""
conditions_to_plot = ['30s',
                      '300s',
                      ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
figax_learned_trial = ff.plotBarsFromDict(learned_trial_lick,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to learn',
                                           save_stats = save_stats,
                                           logscale = True,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure3_ExtDataFig6ae,
                                           )
#%%
"""
FIGURE 3:
 - PANEL F: LICK TIMECOURSE 30 60 300 600 SCALED TRIAL UNITS
"""
plot_lick = True
scaled = True
scaled_align_to_1 = True
linewidth_0_lick = 1
linewidth_lick = 0.35
ylim_lick = [None, 6]
plot_cumsum = False
fig_lick_avg_trials_both_scaled, ax_lick_avg_trials_both_scaled = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                                        conditions_to_plot = ['60s',
                                                                                                              '600s',
                                                                                                              '300s',
                                                                                                              '30s',],
                                                                                        colors_for_conditions = dc.colors_for_conditions,
                                                                                        plot_cumsum = plot_cumsum,
                                                                                        plot_lick = plot_lick,
                                                                                        linewidth_lick = linewidth_lick,
                                                                                        linewidth_0_lick = linewidth_0_lick,
                                                                                        ylim_lick = ylim_lick,
                                                                                        scaled = scaled,
                                                                                        scaled_align_to_1 =scaled_align_to_1,
                                                                                        axsize = dc.axsize_timecourse,
                                                                                        save_figs = save_figs,
                                                                                        fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                                                        )
#%%
"""
FIGURE 3:
 - PANEL G: TRIALS TO LEARN VS IRI SCATTER WITH FIT LINE
 - PANEL H: TOTAL TIME TO LEARN 30 60 300 600s
"""
conditions_to_plot = ['30s',
                      '60s',
                      '300s',
                      '600s',]
ylim = [None, 320]
xlim = [10, 1001]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
IRI_vs_trials_to_learn = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                      colors_for_conditions = dc.colors_for_conditions,
                                      conditions_to_plot = conditions_to_plot,
                                      xlim = xlim,
                                      ylim = ylim,
                                      axsize = dc.axsize_timecourse,
                                      save_fig = save_figs,
                                      fig_path = fig_path_Figure3_ExtDataFig6ae,
                                      )
#test if slope is different from -1
hypothesized_slope = -1
pval_slope, tstat_slope = sf.compare_linregress_slope_onesample(IRI_vs_trials_to_learn['fit_line'],
                                                                len(conditions_to_plot),
                                                                hypothesized_slope = hypothesized_slope,
                                                                )
print(f"Regression slope {IRI_vs_trials_to_learn['fit_line'].slope} vs. hypothesized {hypothesized_slope}: t = {tstat_slope}, p = {pval_slope}")


time_to_learn_dict = lpf.calculate_time_to_learn_from_learned_trials(learned_trial_lick, df_behavior_trials_CSplus_learners)
figax_learned_trial = ff.plotBarsFromDict(time_to_learn_dict,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='total conditioning time before cue licking (s)',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_4,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure3_ExtDataFig6ae,
                                           )
#stats
time_to_learn_anova = sf.one_way_anova_from_dict(time_to_learn_dict,
                            label_key = 'condition',
                            label_values = 'time_to_learn',
                            keys_to_include = conditions_to_plot,
                            assume_equal_variance = False,
                            save_stats = True,
                            fig_path = fig_path_Figure3_ExtDataFig6ae)

#%%
"""
EXTENDED DATA FIGURE 6:
 - PANEL A: ZOOM OF FIRST 80 TRIALS 30 300s TIMECOURSE
 - PANEL B: BAR GRAPH COMPARING LICK RATES TRIALS 71 - 80 FOR 30s AND 300s
 - PANEL C: INDIVIDUAL CUMSUM PLOTS 30s 300s
 - PANEL D: BAR GRAPH COMPARING ASYMPOTITIC LICK RATES 30s AND 300s
 - PANEL E: 60s and 600s SCATTER IRI VS TRIALS TO LEARN DA WITH FIT LINE FOR
"""
conditions_to_plot = ['30s',
                      '300s',
                      ]
#PANEL A
plot_lick = True
fig_lick_avg_trials_both_first80, ax_lick_avg_trials_both_first80 = ff.plotDALickOverTime(df_first_80_trials,
                                                                                          conditions_to_plot = conditions_to_plot,
                                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                                          plot_lick = plot_lick,
                                                                                          axsize = dc.axsize_timecourse,
                                                                                          save_figs = save_figs,
                                                                                          fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                                                          )
#PANEL B
plot_lick_or_DA = 'lick'
range_to_plot = {'30s': [71, 80],
                 '300s': [71, 80]}
fig_lickratemeans, ax_lickratemeans = ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                               conditions_to_plot = conditions_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               plot_lick_or_DA = plot_lick_or_DA,
                                                               ylabel ='lick rate to cue (Hz)\ntrials 71 - 80',
                                                               save_stats = save_stats,
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                               ylim = [None, None],
                                                               axsize = dc.axsize_bars_2
                                                               )
#PANEL C
supplment_all_60s600s_cumsum = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus,
                                                                  conditions_to_plot = conditions_to_plot,
                                                                  colors_for_conditions = dc.colors_for_conditions,
                                                                  linewidth_learned_trial = 0.25,
                                                                  plot_all_individuals = True,
                                                                  plot_examples = False,
                                                                  condition_examples = dc.cumsum_examples_behavior,
                                                                  nonlearners_list = nonlearners_list,
                                                                  axsize = dc.axsize_cumsum_all_individuals,
                                                                  sharex = 'row',
                                                                  save_fig = save_figs,
                                                                  fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                                  )
#PANEL D
plot_lick_or_DA = 'lick'
conditions_to_plot = ['30s',
                      '300s',
                      ]
range_to_plot = {'30s': [601, 800],
                 '300s': [61, 80]}
fig_lickratemeans, ax_lickratemeans = ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                               conditions_to_plot = conditions_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               plot_lick_or_DA = plot_lick_or_DA,
                                                               ylabel ='lick rate to cue (Hz)\nlast 20 or 200 trials',
                                                               save_stats = save_stats,
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure3_ExtDataFig6ae,
                                                               ylim = [None, None],
                                                               axsize = dc.axsize_bars_2
                                                               )
#PANEL E
conditions_to_plot = ['60s',
                      '600s',
                      ]

xlim = [10, 1000]
ylim = [1, 100]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      get_DA_learned_trial = True,
                                                      )
learned_trial_DA = learned_trial_data['learned_trial_DA']
IRI_vs_trials_to_learn_DA = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_DA,
                                      colors_for_conditions = dc.colors_for_conditions,
                                      conditions_to_plot = conditions_to_plot,
                                      conditions_for_fitline = conditions_to_plot,
                                      xlim = xlim,
                                      ylim = ylim,
                                      ax_to_plot = None,
                                      axsize = dc.axsize_timecourse,
                                      save_fig = save_figs,
                                      fig_path = fig_path_Figure3_ExtDataFig6ae
                                      )