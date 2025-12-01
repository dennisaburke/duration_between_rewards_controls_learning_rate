# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

FIGURE 1: Behavioral learning in one-tenth the experiences with ten times the trial spacing.

"""
"""
imports
"""
import os

import functions.load_preprocess  as lp
import functions.default_configs as dc
import functions.figure_functions as ff
import functions.lick_photo_functions as lpf

#%%
"""
set paths to data and outputs
"""
#set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

nwb_dir_path = r'..\DATA_experimental\001632'
figure_path_root = r'..\FIGURES'
fig_path_Figure1_ExtDataFig12 = os.path.join(figure_path_root, r'Figure1_ExtDataFig12')
save_figs = False
save_stats = False
#%%
"""
load and prepare data
"""

nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,  ['600s', '60s'])
#%%
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )
df_behavior_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)
df_behavior_days_CSplus = lp.get_behavior_days_CSplus_df(df)
nonlearners_list = lpf.get_nonlearners(df_behavior_days_CSplus)
df_first_40_trials = df_behavior_trials_CSplus[df_behavior_trials_CSplus['cue_trial_num'] <=40]

#%%


"""
FIGURE 1:
 - PANEL D: EXAMPLE LICK RASTER AND PSTH FOR EXAMPLE 60S AND 600S MICE
"""

plot_lick = True
plot_DA = False

fig_ex_behavior_600s, ax_ex_behavior_600s  = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_behavior['600s'],
                                                            color_lick_PSTH = dc.colors_for_conditions['600s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            axsize = dc.axsize_raster_PSTH_panel,
                                                            fig_path = fig_path_Figure1_ExtDataFig12,
                                                            save_fig = save_figs,
                                                            )
fig_ex_behavior_60s, ax_ex_behavior_60s  = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_behavior['60s'],
                                                            color_lick_PSTH = dc.colors_for_conditions['60s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            axsize = dc.axsize_raster_PSTH_panel,
                                                            fig_path = fig_path_Figure1_ExtDataFig12,
                                                            save_fig = save_figs,
                                                            )
#%%
"""
FIGURE 1:
 - PANEL E: CUE LICKING TIMECOURSE AND ZOOMED IN TIMECOURSE 60 600S
"""
plot_lick = True
fig_lick_avg_trials_both, ax_lick_avg_trials_both = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                          conditions_to_plot = ['600s',
                                                                                                '60s'],
                                                                          colors_for_conditions =dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          ylim_lick = [None, 5],
                                                                          xlim_lick = [1, 400],
                                                                          fig_path  = fig_path_Figure1_ExtDataFig12,
                                                                          )
#zoom of first 40 trials
fig_lick_avg_trials_both_first40, ax_lick_avg_trials_both_first40 = ff.plotDALickOverTime(df_first_40_trials,
                                                                                          conditions_to_plot = ['600s',
                                                                                                                '60s'],
                                                                                          colors_for_conditions =dc.colors_for_conditions,
                                                                                          plot_lick = plot_lick,
                                                                                          axsize = dc.axsize_timecourse_inset,
                                                                                          save_figs = save_figs,
                                                                                          fig_path = fig_path_Figure1_ExtDataFig12,
                                                                                          )
#%%
"""
FIGURE 1:
 - PANEL F: CUE LICK CUMSUM WITH LEARNED TRIAL EXAMPLE 60 AND 600S ITI
"""
linewidth_lick = 1.5
example_cumsum_lick_600s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus_learners,
                                                                animal = dc.cumsum_examples_behavior['600s'],
                                                                color_lick = dc.colors_for_conditions['600s'],
                                                                linewidth_lick = linewidth_lick,
                                                                axsize = dc.axsize_cumsum_examples,
                                                                save_fig =save_figs,
                                                                fig_path = fig_path_Figure1_ExtDataFig12,
                                                                )
example_cumsum_lick_60s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus_learners,
                                                                animal = dc.cumsum_examples_behavior['60s'],
                                                                color_lick = dc.colors_for_conditions['60s'],
                                                                linewidth_lick = linewidth_lick,
                                                                axsize = dc.axsize_cumsum_examples,
                                                                save_fig =save_figs,
                                                                fig_path = fig_path_Figure1_ExtDataFig12,
                                                                )
#%%
"""
FIGURE 1:
 - PANEL G: TRIALS TO LEARN 60 AND 600S BARGRAPHS WITH MEAN AND VARIANCE TEST

"""
conditions_to_plot = ['600s',
                      '60s']
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot= conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
order_to_plot = ['60s', '600s']
figax_learned_trial = ff.plotBarsFromDict(learned_trial_lick,
                                           condition_colors = dc.colors_for_conditions,
                                           order_to_plot = order_to_plot,
                                           ylabel ='trials to learn',
                                           save_stats = save_stats,
                                           logscale = True,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure1_ExtDataFig12,
                                           )

#%%
"""
FIGURE 1:
 - PANEL H: TIME TO LEARN 60 AND 600s

"""
conditions_to_plot = ['60s',
                      '600s',
                      ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
time_to_learn_dict = lpf.calculate_time_to_learn_from_learned_trials(learned_trial_lick, df_behavior_trials_CSplus_learners)


figax_learned_trial = ff.plotBarsFromDict(time_to_learn_dict,
                                           condition_colors = dc.colors_for_conditions,
                                           order_to_plot = conditions_to_plot,
                                           ylabel ='total conditioning time before cue licking (s)',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure1_ExtDataFig12,
                                           )
#%%
"""
FIGURE 1:
 - PANEL I & J: SCALED TIMECOURSES 60 AND 600s AVERAGE LICK AND CUMSUM
"""
plot_lick = True
scaled = True
scaled_align_to_1 = True
linewidth_0_lick = 1
conditions_to_plot = ['600s',
                      '60s']
ylim = [None, 5]
fig_lick_avg_trials_both_scaled, ax_lick_avg_trials_both_scaled = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          linewidth_0_lick = linewidth_0_lick,
                                                                          scaled = scaled,
                                                                          ylim_lick = ylim,
                                                                          axsize = dc.axsize_timecourse,
                                                                          scaled_align_to_1 = scaled_align_to_1,
                                                                          save_figs = save_figs,
                                                                          fig_path = fig_path_Figure1_ExtDataFig12,
                                                                          )
plot_cumsum = True
plot_lick = True
linewidth_lick = 2
scaled = True
shaded_error = False
plot_error = False
plot_individuals = True
alpha_individuals = 0.6

fig_lick_cum_trials_both, ax_lick_cum_trials_both = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_cumsum = plot_cumsum,
                                                                          plot_lick = plot_lick,
                                                                          scaled = scaled,
                                                                          scaled_align_to_1 =scaled_align_to_1,
                                                                          linewidth_lick = linewidth_lick,
                                                                          plot_error = plot_error,
                                                                          plot_individuals = plot_individuals,
                                                                          alpha_individuals = alpha_individuals,
                                                                          linewidth_0_lick = linewidth_0_lick,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          fig_path = fig_path_Figure1_ExtDataFig12,
                                                                          )
#%%
"""
FIGURE 1:
 - PANEL K: ASYMPTOTIC LICK RATE BAR GRAPH 60 AND 600S
"""
conditions_to_plot = ['60s',
                      '600s',]
range_to_plot = {'600s': [31, 40],
                 '60s': [301, 400]}
fig_lickratemeans, ax_lickratemeans =  ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                               conditions_to_plot = conditions_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               plot_lick_or_DA = 'lick',
                                                               save_stats = save_stats,
                                                               ylim =[None,None],
                                                               ylabel ='lick rate to cue (Hz)\nlast 10 or 100 trials',
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure1_ExtDataFig12,
                                                               )

#%%
"""
EXTENDED DATA FIGURE 1:
 - PANEL A: MORE EXAMPLE LICK RASTER/PSTH 60 AND 600S
 - PANEL B: LICK RATE BAR GRAPH TRIALS 36-40 FOR 60 VS 600S
 - PANEL C: PROBABILITY OF LICK 400 TRIALS ZOOM FIRST 40 TRIALS
 - PANEL D: PROBABILITY OF LICK BAR GRAPH TRIALS 36-40
"""
fig_path_Figure1_ExtDataFig12=''
#PANEL A
plot_lick = True
plot_DA = False
fig_ex_behavior_600s_supp, ax_ex_behavior_600s_supp  = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_behavior_supp['600s'],
                                                            color_lick_PSTH = dc.colors_for_conditions['600s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            fig_path =fig_path_Figure1_ExtDataFig12,
                                                            save_fig = save_figs,
                                                            )

fig_ex_behavior_60s_supp, ax_ex_behavior_60s_supp   = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_behavior_supp['60s'],
                                                            color_lick_PSTH = dc.colors_for_conditions['60s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            fig_path = fig_path_Figure1_ExtDataFig12,
                                                            save_fig = save_figs,
                                                            )

#PANEL B
plot_lick_or_DA = 'lick'
conditions_to_plot = ['60s',
                      '600s',
                      ]
range_to_plot = {'600s': [36, 40],
                 '60s': [36, 40]}
fig_lickratemeans, ax_lickratemeans = ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                               conditions_to_plot = conditions_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               plot_lick_or_DA = plot_lick_or_DA,
                                                               ylabel ='lick rate to cue (Hz)\ntrials 36 - 40',
                                                               save_stats = save_stats,
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure1_ExtDataFig12,
                                                               ylim = [None, None],
                                                               axsize = dc.axsize_bars_2
                                                               )
#PANEL C
prob_antic_lick_plot = ff.plotProbabilityOfAnticLick(df_behavior_trials_CSplus,
                                                     conditions_to_plot = conditions_to_plot,
                                                     colors = dc.colors_for_conditions,
                                                     save_fig = save_figs,
                                                     fig_path = fig_path_Figure1_ExtDataFig12,
                                                     axsize = dc.axsize_timecourse)

prob_antic_lick_plot_zoomed = ff.plotProbabilityOfAnticLick(df_first_40_trials,
                                                             conditions_to_plot = conditions_to_plot,
                                                             colors = dc.colors_for_conditions,
                                                             save_fig = save_figs,
                                                             fig_path = fig_path_Figure1_ExtDataFig12,
                                                             axsize = dc.axsize_timecourse_inset
                                                             )
#PANEL D
pro_antic_lick_bars = ff.compare_prob_antic_lick_bars(df_behavior_trials_CSplus,
                           conditions_to_plot = conditions_to_plot,
                           range_to_plot = range_to_plot,
                           condition_colors =dc.colors_for_conditions,
                           ylabel ='',
                           save_stats = save_stats,
                           save_fig = save_figs,
                           save_png = False,
                           fig_path = fig_path_Figure1_ExtDataFig12,
                           title='',
                           maxNLocator = True,
                           axsize = (0.64, 1),
                           )
#%%
"""
EXTENDED DATA FIGURE 2:
 - PANEL A: EXAMPLE 60S CUMSUM, NORMED CUMSUM AND AVERAGE LICK RATE W/LEARNED TRIAL
 - PANEL B: EXAMPLE 600S CUMSUM, NORMED CUMSUM AND AVERAGE LICK RATE W/LEARNED TRIAL
 - PANEL C: ALL 60 AND 600S INDIVIUDAL ANIMAL LICK CUMSUM W/LEARNED TRIAL
"""

norm_lick_left = True
plot_cumsum = True
plot_norm_and_raw = True
ylim_left = [None, 6]
linewidth_diagonal_lick = 0.5

#A & B CUMSUM PLOTS
example_cumsum_lick_600s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus,
                                                                animal = dc.cumsum_examples_behavior['600s'],
                                                                color_lick = dc.colors_for_conditions['600s'],
                                                                linewidth_diagonal_lick = linewidth_diagonal_lick,
                                                                ylim_left = ylim_left,
                                                                plot_norm_and_raw = plot_norm_and_raw,
                                                                norm_lick_left =norm_lick_left,
                                                                axsize = dc.axsize_cumsum_examples,
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure1_ExtDataFig12,
                                                                )
example_cumsum_lick_60s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus,
                                                                animal = dc.cumsum_examples_behavior['60s'],
                                                                color_lick = dc.colors_for_conditions['60s'],
                                                                linewidth_diagonal_lick = linewidth_diagonal_lick,
                                                                ylim_left = ylim_left,
                                                                plot_norm_and_raw = plot_norm_and_raw,
                                                                norm_lick_left =norm_lick_left,
                                                                axsize = dc.axsize_cumsum_examples,
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure1_ExtDataFig12,
                                                                )

#A & B AVERAGE PLOTS
plot_cumsum = False
linewidth_diagonal_lick = 0.5
linewidth_lick = 0.35

example_cumsum_lick_600s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus,
                                                                animal = dc.cumsum_examples_behavior['600s'],
                                                                color_lick = dc.colors_for_conditions['600s'],
                                                                plot_cumsum = plot_cumsum,
                                                                linewidth_lick = linewidth_lick,
                                                                linewidth_diagonal_lick = linewidth_diagonal_lick,
                                                                axsize = dc.axsize_cumsum_examples,
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure1_ExtDataFig12,
                                                                )
linewidth_lick = 0.5
example_cumsum_lick_60s = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus,
                                                                animal = dc.cumsum_examples_behavior['60s'],
                                                                color_lick = dc.colors_for_conditions['60s'],
                                                                plot_cumsum = plot_cumsum,
                                                                linewidth_lick = linewidth_lick,
                                                                linewidth_diagonal_lick = linewidth_diagonal_lick,
                                                                axsize = dc.axsize_cumsum_examples,
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure1_ExtDataFig12,
                                                                )
#PANEL C
conditions = ['60s', '600s']
supplment_all_60s600s_cumsum = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus,
                                                                  conditions_to_plot = conditions,
                                                                  colors_for_conditions = dc.colors_for_conditions,
                                                                  linewidth_learned_trial = 0.25,
                                                                  plot_all_individuals = True,
                                                                  plot_examples = False,
                                                                  nonlearners_list = nonlearners_list,
                                                                  condition_examples = dc.cumsum_examples_behavior,
                                                                  axsize = dc.axsize_cumsum_all_individuals,
                                                                  plot_on_2_lines = True,
                                                                  sharex = 'row',
                                                                  save_fig = save_figs,
                                                                  fig_path = fig_path_Figure1_ExtDataFig12,
                                                                  )
#%%
"""
EXTENDED DATA FIGURE 2:
 - PANEL D: BAR GRAPHS FOR DIFFERENT % MAX DIST LEARNED TRIAL CALCULATION
"""
conditions = ['60s', '600s']
#trials to learn calculation with different % max dists from diagonal
learned_trial_distances = [1, 0.95, 0.90, 0.85, 0.80]
for dist in learned_trial_distances:
    learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                          conditions_to_plot = conditions,
                                                          percent_max_dist = dist,
                                                          )
    learned_trial_lick = learned_trial_data['learned_trial_lick']
    order_to_plot = ['60s', '600s']
    figax_learned_trial = ff.plotBarsFromDict(learned_trial_lick,
                                               condition_colors = dc.colors_for_conditions,
                                               order_to_plot = order_to_plot,
                                               ylabel ='trials to learn',
                                               title = str(dist).replace('.', ''),
                                               data_is_nested_dict = True,
                                               save_stats = save_stats,
                                               logscale = True,
                                               save_fig = save_figs,
                                               fig_path = fig_path_Figure1_ExtDataFig12,
                                               )
#%%
"""
EXTENDED DATA FIGURE 2:
 - PANEL E: LICK RATE ACROSS DAYS 60 VS 600S
 - PANEL F: PROBABILITY OF LICK SCALED TRIALS
 - PANEL G: PROBABILITY OF LICK ASYMPTOTE
 - PANEL H: ABRUPTNESS OF CHANGE AT LEARNING
"""
conditions_to_plot = ['600s',
                      '60s',
                      ]
plot_days = True
plot_cumsum = False
plot_lick = True
linewidth_lick = 0.5
plot_days = True
shaded_error = False
plot_symbols = True
linewidth_0_lick = 1
#PANEL E
fig_days, ax_days = ff.plotDALickOverTime(df_behavior_days_CSplus,
                                        conditions_to_plot = conditions_to_plot,
                                        colors_for_conditions = dc.colors_for_conditions,
                                        plot_cumsum = plot_cumsum,
                                        plot_lick = plot_lick,
                                        plot_days = plot_days,
                                        shaded_error = shaded_error,
                                        plot_symbols = plot_symbols,
                                        linewidth_lick = linewidth_lick,
                                        linewidth_0_lick = linewidth_0_lick,
                                        axsize = dc.axsize_timecourse,
                                        save_figs = save_figs,
                                        fig_path = fig_path_Figure1_ExtDataFig12,
                                        )
#PANEL F
prob_antic_lick_plot = ff.plotProbabilityOfAnticLick(df_behavior_trials_CSplus,
                                                     conditions_to_plot = conditions_to_plot,
                                                     linewidth_0_lick = linewidth_0_lick,
                                                     scaled = True,
                                                     colors = dc.colors_for_conditions,
                                                     save_fig = save_figs,
                                                     fig_path = fig_path_Figure1_ExtDataFig12,
                                                     axsize = dc.axsize_timecourse,
                                                     )
#PANEL G
conditions_to_plot = ['60s',
                      '600s',
                      ]
range_to_plot = {'600s': [31, 40],
                 '60s': [301, 400]}
antic_lick_threshold = 2
pro_antic_lick_bars = ff.compare_prob_antic_lick_bars(df_behavior_trials_CSplus,
                           conditions_to_plot = conditions_to_plot,
                           range_to_plot = range_to_plot,
                           condition_colors = dc.colors_for_conditions,
                           antic_lick_threshold = antic_lick_threshold,
                           ylabel ='',
                           save_stats = save_stats,
                           save_fig = save_figs,
                           save_png = False,
                           fig_path = fig_path_Figure1_ExtDataFig12,
                           title='',
                           maxNLocator = True,
                           axsize = (0.64, 1),
                           )

#PANEL H
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )

abruptness = learned_trial_data['abruptness_learned_trial_lick_norm']
order_to_plot = ['60s', '600s']
figax_learned_trial = ff.plotBarsFromDict(abruptness,
                                           condition_colors = dc.colors_for_conditions,
                                           order_to_plot = order_to_plot,
                                           ylabel ='abruptness of change',
                                           save_stats = save_stats,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure1_ExtDataFig12,
                                           )