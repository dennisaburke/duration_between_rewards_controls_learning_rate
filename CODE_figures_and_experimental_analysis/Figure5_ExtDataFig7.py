# -*- coding: utf-8 -*-
"""
Created on Sun May  4 18:09:58 2025

@author: DeBurke
"""

"""
FIGURE 5
"""


"""
imports
"""
import os
import pingouin
import scipy.stats as stats
import statsmodels
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from natsort import natsorted

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
fig_path_Figure5_ExtDataFig7 = os.path.join(figure_path_root, r'Figure5_ExtDataFig7')

save_figs = False
save_stats = False

#%%
"""
load and prepare experimental data

"""


nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,  ['30s', '60s', '300s', '600s', '3600s'])
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )

df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)

#%%
"""
LOAD BEHAVIOR_FIT SUMMARY FILE OUTPUT TO GET TRIALS TO LEARN FOR EACH BEST FIT MODEL COMBINATION

(SUMMARY FILES FROM OUR SIMULATION RUN ON GITHUB IN FOLDERS BELOW SO THIS CAN BE RERUN WITHOUT RESIMULATING ANYTHING)
"""
#set paths to model data and summary data (generated from 'behavior_fit' matlab script in CODE_simulations directory)
path_summarydata = r'..\DATA_simulation_outputs\behavior_fit_summary_files\modelfit_summarydata_bestfit.mat'
path_summarydata_microstim400trials = r'..\DATA_simulation_outputs\modelfit_summarydata_bestfit400trialsPerSessionMicrostim.mat'
path_data={'microstim': r'..\DATA_simulation_outputs\microstimulus_sweep',
           'sop': r'..\DATA_simulation_outputs\sop_sweep',
           'anccr': r'..\DATA_simulation_outputs\anccr_sweep',
           'microstim_scaling': r'..\DATA_simulation_outputs\microstimulus_scaling_sweep', #model57 from microstim run
           'sop_unconstrained': r'..\DATA_simulation_outputs\sop_sweep',
           }
modelfit_summarydata = sio.loadmat(path_summarydata, simplify_cells = True)
labels_model = ['microstim', 'sop', 'anccr', 'microstim_scaling', 'sop_unconstrained']
labels_ITI = ['30 s ITI', '60 s ITI', '300 s ITI', '600 s ITI', '3600 s ITI',]
num_trials_by_ITI = [800, 400, 88, 48, 16]
#num_trials_by_ITI = [800, 400, 400, 400, 400]
# num_trials_by_ITI = [2400, 1200, 264, 144, 48]
trials_to_analyze_by_ITI = [-200, -100, -20, -10, -3] # for asymptotes
num_iterations_per_ITI = 20
len_trial_in_s = 1.25

#get best fit model idxs from data
idx_best_fit_model = {}
best_fit_models_learned_trials_array_dict ={}

for model in labels_model:
    if model  in ['microstim', 'microstim_scaling', 'sop', 'sop_unconstrained'] :
        idx_best_fit_model[model] = modelfit_summarydata[f'{model}_best_fit_model_num']
    else:
        idx_best_fit_model[model] = modelfit_summarydata[f'{model}_best_fit_model_idx']
    best_fit_models_learned_trials_array_dict[model] = modelfit_summarydata[f'{model}_best_fit_TTL']
#unpack simulation result of learned trials so in same format as behavior data learned trials
best_fit_models_learned_trials_by_ITI = {mod: {iti: learned_trls[x]
                                               for (x, iti)
                                               in enumerate(labels_ITI)
                                               }
                                         for (mod, learned_trls)
                                         in best_fit_models_learned_trials_array_dict.items()
                                         }
#%%
"""
FIGURE 5:
 - PANEL B: CALCULATE TOTAL TIME UNTIL CUE LICKING FOR 30 60 30 600 3600
     + GET SLOPE OF REGRESSION THROUGH POINTS (FOR EXT DATA FIG 7A)
"""

conditions_to_plot = ['30s',
                      '60s',
                      '300s',
                      '600s',
                      '3600s',
                      ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
time_to_learn_dict = lpf.calculate_time_to_learn_from_learned_trials(learned_trial_lick,
                                                                     df_behavior_trials_CSplus_learners,
                                                                     )
condition_ITIs = {'30s': 30,
                  '30 s ITI': 30,
                  '60s': 60,
                  '60 s ITI': 60,
                  '300s': 300,
                  '300 s ITI': 300,
                  '600s': 600,
                  '600 s ITI': 600,
                  '3600s': 3600,
                  '3600 s ITI': 3600,
                  }
x_list = []
y_list = []
for condition in time_to_learn_dict.keys():
    if condition_ITIs[condition] != 3600:
        num_animals = len(time_to_learn_dict[condition].keys())
        x_condition = [condition_ITIs[condition]] *num_animals
        x_list =  x_list + x_condition
        y_condition = list(time_to_learn_dict[condition].values())
        y_list = y_list + y_condition
x_array = np.array(x_list)
y_array = np.array(y_list)

time_to_learn_regression_behav = stats.linregress(x_array, y_array)
time_to_learn_regression_behav_n = len(x_array)
plt.figure()
plt.scatter(x_array, y_array)
xaxis = np.linspace(30, 600, 100)
plt.plot(xaxis, time_to_learn_regression_behav.slope*xaxis + time_to_learn_regression_behav.intercept)


time_to_learn_all_behavior = ff.plotBarsFromDict_brokenAxis(time_to_learn_dict,
                                                            condition_colors = dc.colors_for_conditions,
                                                            ylabel = 'total conditioning time before cue licking (s) broken y',
                                                            height_ratios=[1, 6],
                                                            ylim1 = [14000, 19000],
                                                            ylim2 = [0, 12000],
                                                            MultipleLocator = 5000,
                                                            axsize = dc.axsize_bars_4,
                                                            save_fig = save_figs,
                                                            fig_path = fig_path_Figure5_ExtDataFig7,
                                                            )


#%%
"""
FIGURE 5 & EXTENDED DATA FIGURE 7:
 - FIGS 5E, 5H, 5K, & EXT DATA FIGS 7D, 7F: TOTAL TIME TO LEARN
 - CALCULATE TOTAL TIME UNTIL BEHAVIOR FOR BEST FIT MODELS FOR TDRL (FIG 5E), SOP (FIG 5H), ANCCR (FIG 5K),
   TDRL WITH ALPHA SCALED BY IRI (EXT DATA FIG 7D), AND SOP WITHOUT PD1>PD2 CONSTRAINT (EXT DATA FIG 7)
     + GET SLOPE OF REGRESSION THROUGH POINTS (FOR EXT DATA FIG 7A)
"""
#calculate time to learn by for each iteration of each model based on trials to learn and mean ITI
time_to_learn_models_dict = dict.fromkeys(best_fit_models_learned_trials_by_ITI.keys(),[])
regression_models_dict = dict.fromkeys(best_fit_models_learned_trials_by_ITI.keys(),[])
regression_models_n_dict = dict.fromkeys(best_fit_models_learned_trials_by_ITI.keys(),[])
for model in best_fit_models_learned_trials_by_ITI.keys():
    time_to_learn_models_dict[model] = dict.fromkeys(best_fit_models_learned_trials_by_ITI[model].keys(),[])
    for iti in best_fit_models_learned_trials_by_ITI[model].keys():
        total_ITI_time = float(iti.split()[0]) * (best_fit_models_learned_trials_by_ITI[model][iti]+1)
        total_trial_time = 4.25 * best_fit_models_learned_trials_by_ITI[model][iti]
        total_time_to_learn = total_ITI_time + total_trial_time
        time_to_learn_models_dict[model][iti] = total_time_to_learn
axlims = {'microstim': [60000, 62000],
          'microstim_scaling': [14000,16000],
          'anccr': [14000,16000],
          'sop': [46000, 48000],
          'sop_unconstrained': [42000, 44000],
          }
for model in time_to_learn_models_dict.keys():
    time_to_learn_simulation = ff.plotBarsFromDict_brokenAxis(time_to_learn_models_dict[model],
                                                   condition_colors = dc.colors_for_conditions,
                                                   ylabel =f'{model} total time to learn (s) brokeny',
                                                   data_is_nested_dict = False,
                                                   data_is_regular_dict = True,
                                                   axsize = dc.axsize_bars_4,
                                                   ylim1 = axlims[model],
                                                   ylim2 = [0, 12000],
                                                   save_fig = save_figs,
                                                   fig_path = fig_path_Figure5_ExtDataFig7,
                                                   )
    condition_ITIs = {'30s': 30,
                      '30 s ITI': 30,
                      '60s': 60,
                      '60 s ITI': 60,
                      '300s': 300,
                      '300 s ITI': 300,
                      '600s': 600,
                      '600 s ITI': 600,
                      '3600s': 3600,
                      '3600 s ITI': 3600,
                      }
    x_list = []
    y_list = []
    for condition in time_to_learn_models_dict[model].keys():
        if condition_ITIs[condition] != 3600:
            num_animals = len(time_to_learn_models_dict[model][condition])
            x_condition = [condition_ITIs[condition]] *num_animals
            x_list =  x_list + x_condition
            y_condition = list(time_to_learn_models_dict[model][condition])
            y_list = y_list + y_condition
    x_array = np.array(x_list)
    y_array = np.array(y_list)

    time_to_learn_regression = stats.linregress(x_array, y_array)
    regression_models_dict[model] = time_to_learn_regression
    regression_models_n_dict[model] = len(x_array)
    print(f'{model} time to learn slope 30-600s = {time_to_learn_regression.slope}')
    plt.figure()
    plt.scatter(x_array, y_array)
    xaxis = np.linspace(30, 600, 100)
    plt.plot(xaxis, time_to_learn_regression.slope*xaxis + time_to_learn_regression.intercept)
    plt.title(model)

#%%
"""
EXTENDED DATA FIGURE 7:
 - PANEL A: COMPARISON OF REGRESSION SLOPES FROM 30 - 600S ALL MODELS VS. EXPERIMENTAL
"""
#compare slopes to behavior
pval_list = []
tstat_list = []
bar_fig, bar_ax = plt.subplots(figsize =(2,3))
bar_ax.axhline(y = 0,
                color ='black',
                linestyle=(0,(4,2)),
                linewidth =  0.25,
                alpha = 1,
                )
bar_ax.bar(0, time_to_learn_regression_behav.slope, yerr = time_to_learn_regression_behav.stderr, width = 0.8,
           color = 'black', linewidth = 0, alpha = 0.3, ecolor = 'k',
           error_kw = {'elinewidth': 0.5})
xtick_labels = ['experimental']
for m_idx, model in enumerate(regression_models_dict.keys()):

    pval, tstat = sf.compare_two_linregress_slopes(time_to_learn_regression_behav,
                                         time_to_learn_regression_behav_n,
                                         regression_models_dict[model],
                                         regression_models_n_dict[model],
                                         two_tailed = True)
    bar_ax.bar(m_idx+1, regression_models_dict[model].slope, yerr = regression_models_dict[model].stderr, width = 0.8,
               color = 'black', linewidth = 0, alpha = 0.3, ecolor = 'k',
               error_kw = {'elinewidth': 0.5})
    tstat_list.append(tstat)
    pval_list.append(pval)
    print(model)
    print(pval)
    xtick_labels.append(model)
bar_ax.set_ylim(None, 20)
bar_ax.set_ylabel('regression slope 30 - 600 s ITI')
bar_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(6)))
bar_ax.set_xticklabels(xtick_labels, rotation = 35, ha ='right')
pf.set_ax_size_inches(dc.axsize_bars_5[0], dc.axsize_bars_5[1], bar_ax, print_new_sizes=False)
pf.standardize_plot_graphics(bar_ax)
fig_path = fig_path_Figure5_ExtDataFig7
if save_figs:
    bar_fig.savefig(os.path.join(fig_path, 'slope_comparisons.pdf'),
                    transparent = True,
                    bbox_inches = 'tight')

corrected_ps = statsmodels.stats.multitest.multipletests(pval_list,
                                                         method= 'fdr_bh',
                                                         )

#%%
"""
EXTENDED DATA FIGURE 7:
 - PANELS B & D-G: PLOT TIME TO LEARN VS IRI FOR BEST EXPERIMENTAL FIT ITERATION OF EACH MODEL ON TOP OF EXPERIMENTAL DATA
     -B: MICROSTIMULUS
     -D: MICROSTIMULUS WITH ALPHA SCALED BY ANCCR RULE
     -E: SOP
     -F: SOP, NO PD1>PD2 CONSTRAINT
     -G: ANCCR
"""
conditions_to_plot = ['30s',
                      '60s',
                      '300s',
                      '600s',
                      '3600s',
                      ]
xlim = [10, 5000]
ylim = [0.9, 320]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']

#PANEL B microstim
IRI_vs_learnedtrial_beh_microstim = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                                               colors_for_conditions = dc.colors_for_conditions,
                                                               plot_line = False,
                                                               error = None,
                                                               xlim = xlim,
                                                               ylim = ylim,
                                                               axsize = dc.axsize_timecourse,
                                                               save_fig = False,
                                                               open_marker = True,
                                                               marker = 'o',
                                                               alpha = 0.5,
                                                               markeredgewidth = 0.75,
                                                               )

IRI_vs_trials_to_learn_microstim = ff.plot_IRI_vs_learned_trial_scatter(best_fit_models_learned_trials_by_ITI['microstim'],
                                                              colors_for_conditions = dc.colors_for_conditions,
                                                              nested_dict = False,
                                                              conditions_to_plot = 'all',
                                                              title = 'microstimulus',
                                                              plot_line = False,
                                                              marker = 'h',
                                                              xlim = xlim,
                                                              ylim = ylim,
                                                              ax_to_plot = IRI_vs_learnedtrial_beh_microstim['ax'],
                                                              axsize = dc.axsize_timecourse,
                                                              save_fig = save_figs,
                                                              fig_path = fig_path_Figure5_ExtDataFig7
                                                              )
#PANEL F sop
IRI_vs_learnedtrial_beh_sop = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                                               colors_for_conditions = dc.colors_for_conditions,
                                                               plot_line = False,
                                                               error = None,
                                                               xlim = xlim,
                                                               ylim = ylim,
                                                               axsize = dc.axsize_timecourse,
                                                               save_fig = False,
                                                               open_marker = True,
                                                               marker = 'o',
                                                               alpha = 0.5,
                                                               markeredgewidth = 0.75,
                                                               )
IRI_vs_learnedtrial_sop = ff.plot_IRI_vs_learned_trial_scatter(best_fit_models_learned_trials_by_ITI['sop'],
                                                              colors_for_conditions = dc.colors_for_conditions,
                                                              nested_dict = False,
                                                              conditions_to_plot = 'all',
                                                              title = 'SOP',
                                                              plot_line = False,
                                                              marker = 's',
                                                              xlim = xlim,
                                                              ylim = ylim,
                                                              ax_to_plot = IRI_vs_learnedtrial_beh_sop['ax'],
                                                              axsize = dc.axsize_timecourse,
                                                              save_fig = save_figs,
                                                              fig_path = fig_path_Figure5_ExtDataFig7
                                                              )

#PANEL G anccr
IRI_vs_learnedtrial_beh_anccr = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                                               colors_for_conditions = dc.colors_for_conditions,
                                                               plot_line = False,
                                                               error = None,
                                                               xlim = xlim,
                                                               ylim = ylim,
                                                               axsize = dc.axsize_timecourse,
                                                               save_fig = False,
                                                               open_marker = True,
                                                               marker = 'o',
                                                               alpha = 0.5,
                                                               markeredgewidth = 0.75,
                                                               )
IRI_vs_learnedtrial_anccr = ff.plot_IRI_vs_learned_trial_scatter(best_fit_models_learned_trials_by_ITI['anccr'],
                                                                  colors_for_conditions = dc.colors_for_conditions,
                                                                  nested_dict = False,
                                                                  conditions_to_plot = 'all',
                                                                  title = 'ANCCR',
                                                                  plot_line = False,
                                                                  marker = 'X',
                                                                  xlim = xlim,
                                                                  ylim = ylim,
                                                                  ax_to_plot = IRI_vs_learnedtrial_beh_anccr['ax'],
                                                                  axsize = dc.axsize_timecourse,
                                                                  save_fig = save_figs,
                                                                  fig_path = fig_path_Figure5_ExtDataFig7
                                                                  )
#PANEL D microstim with alpha scaling
IRI_vs_learnedtrial_beh_ms_scaling = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_line = False,
                                                                error = None,
                                                                xlim = xlim,
                                                                ylim = ylim,
                                                                axsize = dc.axsize_timecourse,
                                                                save_fig = False,
                                                                open_marker = True,
                                                                marker = 'o',
                                                                alpha = 0.5,
                                                                markeredgewidth = 0.75,
                                                                )
IRI_vs_learnedtrial_microstim_scale = ff.plot_IRI_vs_learned_trial_scatter(best_fit_models_learned_trials_by_ITI['microstim_scaling'],
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          nested_dict = False,
                                                                          conditions_to_plot = 'all',
                                                                          title = 'microstimulus with alpha scaling',
                                                                          plot_line = False,
                                                                          marker = 'H',
                                                                          xlim = xlim,
                                                                          ylim = ylim,
                                                                          ax_to_plot = IRI_vs_learnedtrial_beh_ms_scaling['ax'],
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_fig = save_figs,
                                                                          fig_path = fig_path_Figure5_ExtDataFig7,
                                                                          )
#PANEL F sop unconstrained
IRI_vs_learnedtrial_beh_sop_unconstrained = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_line = False,
                                                                error = None,
                                                                xlim = xlim,
                                                                ylim = ylim,
                                                                axsize = dc.axsize_timecourse,
                                                                save_fig = False,
                                                                open_marker = True,
                                                                marker = 'o',
                                                                alpha = 0.5,
                                                                markeredgewidth = 0.75,
                                                                )
IRI_vs_learnedtrial_sop_unconstrained = ff.plot_IRI_vs_learned_trial_scatter(best_fit_models_learned_trials_by_ITI['sop_unconstrained'],
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          nested_dict = False,
                                                                          conditions_to_plot = 'all',
                                                                          title = 'sop_unconstrained',
                                                                          plot_line = False,
                                                                          marker = 's',
                                                                          xlim = xlim,
                                                                          ylim = ylim,
                                                                          ax_to_plot = IRI_vs_learnedtrial_beh_sop_unconstrained['ax'],
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_fig = save_figs,
                                                                          fig_path = fig_path_Figure5_ExtDataFig7,
                                                                          )
#%%
"""
LOAD FULL TIMECOURSE DATA FOR EACH BEST FIT MODEL
NEEDED TO PLOT VALUE/RPE/PA2US/NC TIMECOURSES AND ASYMPTOTIC BAR GRAPHS

(YOU WILL NEED TO RERUN SIMULATIONS IN ORDER TO RUN THE REST OF THIS SCRIPT, MODELS TOO BIG FOR GITHUB)
"""

models_list_all = {}
for model, idx in idx_best_fit_model.items():

        path_to_models = path_data[model]
        list_models = natsorted([f.name
                        for f
                        in os.scandir(path_to_models)
                        if f.is_file()
                        if 'simulation' in f.name])
        models_list_all[model] = list_models
        model_best = [m
                      for m
                      in list_models
                      if (m.split('_')[0].endswith(model[-1]+str(idx))
                          or m.split('_')[0].endswith('s'+str(idx))
                          or m.split('_')[0].endswith('p'+str(idx)))][0]
        if model_best != list_models[idx-1]:
            print(model_best)
            print(list_models[idx-1])
            print(f'Issue with finding data for best {model}')

        if model == 'microstim':
            microstim_rpes = mat73.loadmat(os.path.join(path_to_models, model_best),
                                          use_attrdict=False,
                                          only_include = 'rpes')['rpes']
            microstim_values = mat73.loadmat(os.path.join(path_to_models, model_best),
                                          use_attrdict=False,
                                          only_include = 'values')['values']
            microstim_events = mat73.loadmat(os.path.join(path_to_models, model_best),
                                          use_attrdict=False,
                                          only_include = 'events')['events']
        elif model == 'sop':
            sop_data_all = mat73.loadmat(os.path.join(path_to_models, model_best),
                                                use_attrdict=False,)
            sop_events = sop_data_all['events']

            sop_timeseries_all = sop_data_all['sop_timeseries_all']
            sop_dt = sop_data_all['sop_params']['dt']
            sop_value_all = sop_data_all['sop_value_all']
        elif model == 'microstim_scaling':
            microstim_scaling_rpes = mat73.loadmat(os.path.join(path_to_models, model_best),
                                          use_attrdict=False,
                                          only_include = 'rpes')['rpes']
            microstim_scaling_statesize = mat73.loadmat(os.path.join(path_to_models, model_best),
                                          use_attrdict=False,
                                          only_include = 'microstimparams')['microstimparams']['statesize']
            microstim_scaling_values = mat73.loadmat(os.path.join(path_to_models, model_best),
                                          use_attrdict=False,
                                          only_include = 'values')['values']
            microstim_scaling_events = mat73.loadmat(os.path.join(path_to_models, model_best),
                                          use_attrdict=False,
                                          only_include = 'events')['events']
        elif model == 'anccr':
            anccr_modeldata = mat73.loadmat(os.path.join(path_to_models, model_best),
                                        use_attrdict=False,
                                        only_include = 'modeldata')['modeldata']
            anccr_NCcr = anccr_modeldata['NC_cr']
        elif model == 'sop_unconstrained':
            sop_unconstrained_data_all = mat73.loadmat(os.path.join(path_to_models, model_best),
                                                use_attrdict=False,)
            sop_unconstrained_events = sop_unconstrained_data_all['events']

            sop_unconstrained_timeseries_all = sop_unconstrained_data_all['sop_timeseries_all']
            sop_unconstrained_dt = sop_unconstrained_data_all['sop_params']['dt']
            sop_unconstrained_value_all = sop_unconstrained_data_all['sop_value_all']
#%%
"""
FIGURE 5D & EXTENDED DATA FIGURE 7B:
 - MICROSTIMULUS RPE AND VALUE TIMECOURSES, SCALED AND UNSCALED XAXIS UNITS
"""
num_iterations_per_ITI = 20
num_trials_by_ITI = [800, 400, 400, 400, 400]

value_ITI_trials_dict = {x: {} for x in labels_ITI}
rpe_ITI_trials_dict = {x: {} for x in labels_ITI}
for idx, (iti, num_trials) in enumerate(zip(labels_ITI, num_trials_by_ITI)):
    values_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    values_by_trial[:] = np.nan
    rpe_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    rpe_by_trial[:] = np.nan
    for it_num, (it_rpe, it_val, it_event) in enumerate(zip(microstim_rpes[idx],
                                                  microstim_values[idx],
                                                  microstim_events[idx])):

        cue_idx_all = np.flatnonzero(it_event ==1)
        reward_idx_all = np.flatnonzero(it_event==2)
        value_by_trial = [np.max(it_val[cue:rew])
                          for (cue, rew)
                          in zip (cue_idx_all.astype(int),
                                  reward_idx_all.astype(int))
                          ]
        rpe_by_trial[it_num, :] = it_rpe[cue_idx_all.astype(int)]
        values_by_trial[it_num, :] = value_by_trial
    value_ITI_trials_dict[iti]['mean'] = np.nanmean(values_by_trial, axis = 0)
    value_ITI_trials_dict[iti]['sem'] = stats.sem(values_by_trial, axis = 0, ddof = 1)
    value_ITI_trials_dict[iti]['trials'] = np.arange(len(value_ITI_trials_dict[iti]['mean']))+1

    rpe_ITI_trials_dict[iti]['mean'] = np.nanmean(rpe_by_trial, axis = 0)
    rpe_ITI_trials_dict[iti]['sem'] = stats.sem(rpe_by_trial, axis = 0, ddof = 1)
    rpe_ITI_trials_dict[iti]['trials'] = np.arange(len(rpe_ITI_trials_dict[iti]['mean']))+1
scaled = True
fig_microstim_val, ax_microstim_val = ff.plot_model_sweep_timecourses(value_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'microstimulus - value scaledxaxis',
                                                             ylabel = 'value (cue)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )


fig_microstim_rpe, ax_microstim_rpe = ff.plot_model_sweep_timecourses(rpe_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'microstimulus -RPE scaledxaxis',
                                                             ylabel = 'RPE (cue)',
                                                             ylim = ax_microstim_val.get_ylim(),
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
scaled = False
fig_microstim_val, ax_microstim_val = ff.plot_model_sweep_timecourses(value_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'microstimulus - value',
                                                             ylabel = 'value (cue)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
fig_microstim_rpe, ax_microstim_rpe = ff.plot_model_sweep_timecourses(rpe_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'microstimulus -RPE',
                                                             ylabel = 'RPE (cue)',
                                                             ylim = ax_microstim_val.get_ylim(),
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
#%%
"""
EXTENDED DATA FIGURE 7C:
 - MICROSTIMULUS WITH EXTRA TRIALS (>= 400 TRIALS PER ITI GROUP)
     RPE AND VALUE TIMECOURSES, SCALED AND UNSCALED XAXIS UNITS, RPE ASYMPTOTE BAR GRAPHS
"""
modelfit_summarydata_microstim400trials = sio.loadmat(path_summarydata_microstim400trials, simplify_cells = True)
TDRL_extra_trials_best_model_num = modelfit_summarydata_microstim400trials['microstim_best_fit_model_num']

TTL_to_plot = modelfit_summarydata_microstim400trials['microstim_best_fit_TTL']

TDRL_supp_example_model_TTL = {iti: trls
                          for (iti, trls)
                          in zip(labels_ITI, TTL_to_plot)
                         }

num_trials_by_ITI = [800, 400, 400, 400, 400]


alpha_error = 0.3

conditions_to_plot = ['30s',
                      '60s',
                      '300s',
                      '600s',
                      '3600s',
                      ]
xlim = [10, 5000]
ylim = [0.9, 320]

IRI_vs_learnedtrial_microstim_extra_trials = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                                               colors_for_conditions = dc.colors_for_conditions,
                                                               plot_line = False,
                                                               error = None,
                                                               xlim = xlim,
                                                               ylim = ylim,
                                                               axsize = dc.axsize_timecourse,
                                                               save_fig = False,
                                                               open_marker = True,
                                                               marker = 'o',
                                                               alpha = 0.5,
                                                               markeredgewidth = 0.75,
                                                               )

IRI_vs_trials_to_learn_TDRL = ff.plot_IRI_vs_learned_trial_scatter(TDRL_supp_example_model_TTL,
                                                              colors_for_conditions = dc.colors_for_conditions,
                                                              nested_dict = False,
                                                              conditions_to_plot = 'all',
                                                              title =  'TDRL-extra_trials',
                                                              plot_line = False,
                                                              marker = 'h',
                                                              xlim = xlim,
                                                              ylim = ylim,
                                                              ax_to_plot = IRI_vs_learnedtrial_microstim_extra_trials['ax'],
                                                              axsize = dc.axsize_timecourse,
                                                              save_fig = save_figs,
                                                              fig_path = fig_path_Figure5_ExtDataFig7
                                                              )
model = 'microstim'
path_to_models = path_data[model]
list_models = natsorted([f.name
                        for f
                        in os.scandir(path_to_models)
                        if f.is_file()
                        if 'simulation' in f.name],
                        )
model_best = [m
              for m
              in list_models
              if (m.split('_')[0].endswith(model[-1]+str(TDRL_extra_trials_best_model_num))
                  or m.split('_')[0].endswith('s'+str(TDRL_extra_trials_best_model_num)))
              ][0]
if model_best != list_models[TDRL_extra_trials_best_model_num-1]:
    print(model_best)
    print(list_models[TDRL_extra_trials_best_model_num-1])
    print(f'Issue with finding data for best {model}')


microstim_rpes = mat73.loadmat(os.path.join(path_to_models, model_best),
                              use_attrdict=False,
                              only_include = 'rpes')['rpes']
microstim_values = mat73.loadmat(os.path.join(path_to_models, model_best),
                              use_attrdict=False,
                              only_include = 'values')['values']
microstim_events = mat73.loadmat(os.path.join(path_to_models, model_best),
                              use_attrdict=False,
                              only_include = 'events')['events']
value_ITI_trials_dict = {x: {} for x in labels_ITI}
mean_asymptote_val_by_ITI ={x: [] for x in labels_ITI}
rpe_ITI_trials_dict = {x: {} for x in labels_ITI}
mean_asymptote_rpe_by_ITI = {x: [] for x in labels_ITI}
for idx, (iti, num_trials) in enumerate(zip(labels_ITI, num_trials_by_ITI)):
    values_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    values_by_trial[:] = np.nan
    rpe_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    rpe_by_trial[:] = np.nan
    for it_num, (it_rpe, it_val, it_event) in enumerate(zip(microstim_rpes[idx],
                                                            microstim_values[idx],
                                                            microstim_events[idx]),
                                                        ):
        cue_idx_all = np.flatnonzero(it_event ==1)
        reward_idx_all = np.flatnonzero(it_event ==2)
        value_by_trial = [np.max(it_val[cue:rew+1])
                          for (cue, rew)
                          in zip (cue_idx_all.astype(int),
                                  reward_idx_all.astype(int))
                          ]
        rpe_by_trial[it_num, :] = it_rpe[cue_idx_all.astype(int)]
        values_by_trial[it_num, :] = value_by_trial
        mean_asymptote_val = np.mean(value_by_trial[trials_to_analyze_by_ITI[idx]:])
        mean_asymptote_val_by_ITI[iti].append(mean_asymptote_val)
        mean_asymptote_rpe = np.mean(it_rpe[cue_idx_all.astype(int)][trials_to_analyze_by_ITI[idx]:])
        mean_asymptote_rpe_by_ITI[iti].append(mean_asymptote_rpe)
    value_ITI_trials_dict[iti]['mean'] = np.nanmean(values_by_trial, axis = 0)
    value_ITI_trials_dict[iti]['sem'] = stats.sem(values_by_trial, axis = 0, ddof = 1)
    value_ITI_trials_dict[iti]['trials'] = np.arange(len(value_ITI_trials_dict[iti]['mean']))+1

    rpe_ITI_trials_dict[iti]['mean'] = np.nanmean(rpe_by_trial, axis = 0)
    rpe_ITI_trials_dict[iti]['sem'] = stats.sem(rpe_by_trial, axis = 0, ddof = 1)
    rpe_ITI_trials_dict[iti]['trials'] = np.arange(len(rpe_ITI_trials_dict[iti]['mean']))+1
fig_microstim_val_scale, ax_microstim_val_scale = ff.plot_model_sweep_timecourses(value_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = True,
                                                             title = 'microstimulus - value_extraTrials-scaledX',
                                                             ylabel = 'value (cue)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
fig_microstim_rpe_scale, ax_microstim_rpe_scale = ff.plot_model_sweep_timecourses(rpe_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = True,
                                                             title = 'microstimulus -RPE_extraTrials-scaledX',
                                                             ylabel = 'rpe (cue)',
                                                             ylim = ax_microstim_val_scale.get_ylim(),
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
fig_microstim_val_noscale, ax_microstim_val_noscale = ff.plot_model_sweep_timecourses(value_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = False,
                                                             title = 'microstimulus - value_extraTrials',
                                                             ylabel = 'value (cue)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
fig_microstim_rpe_noscale, ax_microstim_rp_noscalee = ff.plot_model_sweep_timecourses(rpe_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = False,
                                                             title = 'microstimulus -RPE_extraTrials',
                                                             ylabel = 'rpe (cue)',
                                                             ylim = ax_microstim_val_noscale.get_ylim(),
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
#plot asymptotic RPEs
asymptote_RPE = ff.plotBarsFromDict(mean_asymptote_rpe_by_ITI,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel = 'TDRL RPE asymptote _extraTrials',
                                           data_is_nested_dict = False,
                                           data_is_regular_dict = True,
                                           save_fig = save_figs,
                                           axsize = dc.axsize_bars_3,
                                           fig_path = fig_path_Figure5_ExtDataFig7,
                                           )
# #plot asymptotic values (not in paper)
# asymptote_val = ff.plotBarsFromDict(mean_asymptote_val_by_ITI,
#                                            condition_colors = dc.colors_for_conditions,
#                                            ylabel = 'TDRL value asymptote _extraTrials',
#                                            data_is_nested_dict = False,
#                                            data_is_regular_dict = True,
#                                            save_fig = save_figs,
#                                            axsize = dc.axsize_bars_3,
#                                            fig_path = fig_path_Figure5_ExtDataFig7,
#                                            )

#stats to compare 60, 600, and 3600s (groups which have experimental dopamine data)
mean_asymptote_rpe_by_ITI_long_df = sf.convert_dict_to_long_df(mean_asymptote_rpe_by_ITI,
                                                                        label_key = 'condition',
                                                                        label_values = 'rpe',
                                                                        )
kurskal_results_rpe = pingouin.kruskal(data = mean_asymptote_rpe_by_ITI_long_df,
                        dv = 'rpe',
                        between = 'condition',
                        )
labels_ITI_subset = ['60 s ITI', '600 s ITI', '3600 s ITI',]
rpe_mean_subset = sf.convert_dict_to_long_df(mean_asymptote_rpe_by_ITI,
                            label_key = 'condition',
                            label_values = 'rpe',
                            keys_to_include = labels_ITI_subset,
                            )

kruskal_pairwise_tests_mwu_rpe = pingouin.pairwise_tests(data = rpe_mean_subset,
                        dv = 'rpe',
                        between = 'condition',
                        correction = True,
                        padjust = 'bonf',
                        parametric = False,
                        return_desc = True)
MannWhitneyU_60_vs_600  = pingouin.mwu(mean_asymptote_rpe_by_ITI['60 s ITI'], mean_asymptote_rpe_by_ITI['600 s ITI'], alternative='two-sided')
MannWhitneyU_60_vs_3600  = pingouin.mwu(mean_asymptote_rpe_by_ITI['60 s ITI'], mean_asymptote_rpe_by_ITI['3600 s ITI'], alternative='two-sided')
MannWhitneyU_600_vs_3600  = pingouin.mwu(mean_asymptote_rpe_by_ITI['600 s ITI'], mean_asymptote_rpe_by_ITI['3600 s ITI'], alternative='two-sided')
#%%
"""
EXTENDED DATA FIGURE 7D:
 - MICROSTIMULUS WITH ALPHA SCALED BY IRI RPE AND VALUE TIMECOURSES, SCALED XAXIS UNITS

"""
num_trials_by_ITI = [800, 400, 400, 400, 400]
use_model_from_summary_file = False
if not use_model_from_summary_file:
    path_to_file_with_more_trials = r'E:\microstim_scaling_model57_300trialsPerTrialsession\microstimulus_sweep\microstimulus1_simulationdata.mat'
    microstim_scaling_rpes = mat73.loadmat(path_to_file_with_more_trials,
                                  use_attrdict=False,
                                  only_include = 'rpes')['rpes']
    microstim_scaling_values = mat73.loadmat(path_to_file_with_more_trials,
                                  use_attrdict=False,
                                  only_include = 'values')['values']
    microstim_scaling_events = mat73.loadmat(path_to_file_with_more_trials,
                                  use_attrdict=False,
                                  only_include = 'events')['events']
    num_trials_by_ITI = [2400, 2400, 2400, 2400, 2400]

value_ITI_trials_dict = {x: {} for x in labels_ITI}
rpe_ITI_trials_dict = {x: {} for x in labels_ITI}
for idx, (iti, num_trials) in enumerate(zip(labels_ITI, num_trials_by_ITI)):
    values_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    values_by_trial[:] = np.nan
    rpe_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    rpe_by_trial[:] = np.nan
    for it_num, (it_rpe, it_val, it_event) in enumerate(zip(microstim_scaling_rpes[idx],
                                                  microstim_scaling_values[idx],
                                                  microstim_scaling_events[idx])):

        cue_idx_all = np.flatnonzero(it_event ==1)
        reward_idx_all = np.flatnonzero(it_event ==2)
        value_by_trial = [np.max(it_val[cue:rew])
                          for (cue, rew)
                          in zip (cue_idx_all.astype(int),
                                  reward_idx_all.astype(int))
                          ]
        values_by_trial[it_num, :] = value_by_trial
        rpe_by_trial[it_num, :] = it_rpe[cue_idx_all.astype(int)]
    value_ITI_trials_dict[iti]['mean'] = np.nanmean(values_by_trial, axis = 0)
    value_ITI_trials_dict[iti]['sem'] = stats.sem(values_by_trial, axis = 0, ddof = 1)
    value_ITI_trials_dict[iti]['trials'] = np.arange(len(value_ITI_trials_dict[iti]['mean']))+1
    rpe_ITI_trials_dict[iti]['mean'] = np.nanmean(rpe_by_trial, axis = 0)
    rpe_ITI_trials_dict[iti]['sem'] = stats.sem(rpe_by_trial, axis = 0, ddof = 1)
    rpe_ITI_trials_dict[iti]['trials'] = np.arange(len(rpe_ITI_trials_dict[iti]['mean']))+1

scaled = False
fig_microstim_scale_val, ax_microstim_scale_val = ff.plot_model_sweep_timecourses(value_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'microstimulus (scaling) - value (newRun) -scaled',
                                                             ylabel = 'value (cue)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
fig_microstim, ax_microstim = ff.plot_model_sweep_timecourses(rpe_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'microstimulus (scaling) -RPE (newRun) - scaled',
                                                             ylabel = 'RPE (cue)',
                                                             ylim = ax_microstim_scale_val.get_ylim(),
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
#%%
"""
EXTENDED DATA FIGURE 7D:
 -  MICROSTIMULUS WITH ALPHA SCALING EXTRA TRIAL ITERATIONS TO GET ASYMPTOTIC RPE AND VALUE
"""


num_trials_by_ITI = [2400, 2400, 2400, 2400, 2400]
#microstim w/ scaling get last trials for DA measurements
value_ITI_trials_dict = {x: {} for x in labels_ITI}
mean_asymptote_val_by_ITI ={x: [] for x in labels_ITI}
rpe_ITI_trials_dict = {x: {} for x in labels_ITI}
mean_asymptote_rpe_by_ITI = {x: [] for x in labels_ITI}
for idx, (iti, num_trials) in enumerate(zip(labels_ITI, num_trials_by_ITI)):
    values_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    values_by_trial[:] = np.nan
    rpe_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    rpe_by_trial[:] = np.nan
    for it_num, (it_rpe, it_val, it_event) in enumerate(zip(microstim_scaling_rpes[idx],
                                                  microstim_scaling_values[idx],
                                                  microstim_scaling_events[idx])):
        cue_idx_all = np.flatnonzero(it_event ==1)
        reward_idx_all = np.flatnonzero(it_event ==2)
        value_by_trial = [np.max(it_val[cue:rew])
                          for (cue, rew)
                          in zip (cue_idx_all.astype(int),
                                  reward_idx_all.astype(int))
                          ]
        mean_asymptote_val = np.mean(value_by_trial[trials_to_analyze_by_ITI[idx]:])
        mean_asymptote_val_by_ITI[iti].append(mean_asymptote_val)
        mean_asymptote_rpe = np.mean(it_rpe[cue_idx_all.astype(int)][trials_to_analyze_by_ITI[idx]:])
        mean_asymptote_rpe_by_ITI[iti].append(mean_asymptote_rpe)

        values_by_trial[it_num, :] = value_by_trial
        rpe_by_trial[it_num, :] = it_rpe[cue_idx_all.astype(int)]
    value_ITI_trials_dict[iti]['mean'] = np.nanmean(values_by_trial, axis = 0)
    value_ITI_trials_dict[iti]['sem'] = stats.sem(values_by_trial, axis = 0, ddof = 1)
    value_ITI_trials_dict[iti]['trials'] = np.arange(len(value_ITI_trials_dict[iti]['mean']))+1
    rpe_ITI_trials_dict[iti]['mean'] = np.nanmean(rpe_by_trial, axis = 0)
    rpe_ITI_trials_dict[iti]['sem'] = stats.sem(rpe_by_trial, axis = 0, ddof = 1)
    rpe_ITI_trials_dict[iti]['trials'] = np.arange(len(rpe_ITI_trials_dict[iti]['mean']))+1


asymptote_RPE = ff.plotBarsFromDict(mean_asymptote_rpe_by_ITI,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel = 'TDRL RPE asymptote (alpha scaled)',
                                           data_is_nested_dict = False,
                                           data_is_regular_dict = True,
                                           save_fig = save_figs,
                                           axsize = dc.axsize_bars_3,
                                           fig_path = fig_path_Figure5_ExtDataFig7,
                                           )
# #not in paper
# asymptote_val = ff.plotBarsFromDict(mean_asymptote_val_by_ITI,
#                                            condition_colors = dc.colors_for_conditions,
#                                            ylabel = 'TDRL value asymptote (alpha scaled by anccr)',
#                                            data_is_nested_dict = False,
#                                            data_is_regular_dict = True,
#                                            save_fig = save_figs,
#                                            axsize = dc.axsize_bars_3,
#                                            fig_path = fig_path_Figure5_ExtDataFig7,
#                                            )
mean_asymptote_rpe_by_ITI_long_df_scaling = sf.convert_dict_to_long_df(mean_asymptote_rpe_by_ITI,
                                                                        label_key = 'condition',
                                                                        label_values = 'rpe',
                                                                        )
kurskal_results_rpe_scaling = pingouin.kruskal(data = mean_asymptote_rpe_by_ITI_long_df_scaling,
                        dv = 'rpe',
                        between = 'condition')


labels_ITI_subset = ['60 s ITI', '600 s ITI', '3600 s ITI',]
rpe_mean_subset_scaling = sf.convert_dict_to_long_df(mean_asymptote_rpe_by_ITI,
                            label_key = 'condition',
                            label_values = 'rpe',
                            keys_to_include = labels_ITI_subset,
                            )

kruskal_pairwise_tests_mwu_rpe_scaling = pingouin.pairwise_tests(data = rpe_mean_subset_scaling,
                        dv = 'rpe',
                        between = 'condition',
                        correction = True,
                        padjust = 'bonf',
                        parametric = False,
                        return_desc = True)
anova_results_RPE_scaling = stats.f_oneway(*mean_asymptote_rpe_by_ITI.values())

#%%
"""
FIGURE 5G & EXTENDED DATA FIGURE 7E:
 - SOP TIMECOURSES
"""
num_trials_by_ITI = [800, 400, 88, 48, 16]

value_sop_ITI_trials_dict = {x: {} for x in labels_ITI}
pA2US_sop_ITI_trials_dict = {x: {} for x in labels_ITI}
for idx, (iti, num_trials) in enumerate(zip(labels_ITI, num_trials_by_ITI)):
    values_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    values_by_trial[:] = np.nan
    pA2USs_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    pA2USs_by_trial[:] = np.nan
    for it_num, (it_times, it_val, it_event) in enumerate(zip(sop_timeseries_all[idx],
                                                    sop_value_all[idx],
                                                    sop_events[idx])):
        cs_times = it_times[0]['cs_times']
        us_times = it_times[0]['us_times']
        cue_idx_eventlog = np.flatnonzero(it_event ==1)
        reward_idx_eventlog = np.flatnonzero(it_event==2)
        value_by_trial = [np.max(it_val[cue:rew])
                          for (cue, rew)
                          in zip (cue_idx_eventlog.astype(int),
                                  reward_idx_eventlog.astype(int))
                          ]
        pA2US_by_trial = [np.max(it_times[0]['pA2_us'][cue:rew])
                          for (cue, rew)
                          in zip (cue_idx_eventlog,
                                  reward_idx_eventlog)
                          ]
        values_by_trial[it_num, :] = value_by_trial
        pA2USs_by_trial[it_num, :] = pA2US_by_trial
    value_sop_ITI_trials_dict[iti]['mean'] = np.nanmean(values_by_trial, axis = 0)
    value_sop_ITI_trials_dict[iti]['sem'] = stats.sem(values_by_trial, axis = 0, ddof = 1)
    value_sop_ITI_trials_dict[iti]['trials'] = np.arange(len(value_sop_ITI_trials_dict[iti]['mean']))+1
    pA2US_sop_ITI_trials_dict[iti]['mean'] = np.nanmean(pA2USs_by_trial, axis = 0)
    pA2US_sop_ITI_trials_dict[iti]['sem'] = stats.sem(pA2USs_by_trial, axis = 0, ddof = 1)
    pA2US_sop_ITI_trials_dict[iti]['trials'] = np.arange(len(pA2US_sop_ITI_trials_dict[iti]['mean']))+1

scaled = True
fig_sop_val, ax_sop_val = ff.plot_model_sweep_timecourses(value_sop_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'SOP - value',
                                                             ylabel = 'value (cue)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )

fig_sop_pA2US, ax_sop_pA2US = ff.plot_model_sweep_timecourses(pA2US_sop_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'pA2_US',
                                                             ylabel = 'pA2_US',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
scaled = False
fig_sop_pA2US, ax_sop_pA2US = ff.plot_model_sweep_timecourses(pA2US_sop_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'pA2_US - unscaled',
                                                             ylabel = 'pA2_US',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
#%%
"""
EXTENDED DATA FIGURE 7E:
 - UNCONSTRAINED SOP TIMECOURSES
"""
value_sop_unconstrained_ITI_trials_dict = {x: {} for x in labels_ITI}
pA2US_sop_unconstrained_ITI_trials_dict = {x: {} for x in labels_ITI}
for idx, (iti, num_trials) in enumerate(zip(labels_ITI, num_trials_by_ITI)):
    values_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    values_by_trial[:] = np.nan
    pA2USs_by_trial = np.zeros((num_iterations_per_ITI, num_trials))
    pA2USs_by_trial[:] = np.nan

    for it_num, (it_times, it_val, it_event) in enumerate(zip(sop_unconstrained_timeseries_all[idx],
                                                    sop_unconstrained_value_all[idx],
                                                    sop_unconstrained_events[idx])):
        cs_times = it_times[0]['cs_times']
        us_times = it_times[0]['us_times']

        cue_idx_eventlog = np.flatnonzero(it_event ==1)
        reward_idx_eventlog = np.flatnonzero(it_event==2)
        value_by_trial = [np.max(it_val[cue:rew])
                          for (cue, rew)
                          in zip (cue_idx_eventlog.astype(int),
                                  reward_idx_eventlog.astype(int))
                          ]
        pA2US_by_trial = [np.max(it_times[0]['pA2_us'][cue:rew])
                          for (cue, rew)
                          in zip (cue_idx_eventlog,
                                  reward_idx_eventlog)
                          ]
        values_by_trial[it_num, :] = value_by_trial
        pA2USs_by_trial[it_num, :] = pA2US_by_trial
    value_sop_unconstrained_ITI_trials_dict[iti]['mean'] = np.nanmean(values_by_trial, axis = 0)
    value_sop_unconstrained_ITI_trials_dict[iti]['sem'] = stats.sem(values_by_trial, axis = 0, ddof = 1)
    value_sop_unconstrained_ITI_trials_dict[iti]['trials'] = np.arange(len(value_sop_unconstrained_ITI_trials_dict[iti]['mean']))+1
    pA2US_sop_unconstrained_ITI_trials_dict[iti]['mean'] = np.nanmean(pA2USs_by_trial, axis = 0)
    pA2US_sop_unconstrained_ITI_trials_dict[iti]['sem'] = stats.sem(pA2USs_by_trial, axis = 0, ddof = 1)
    pA2US_sop_unconstrained_ITI_trials_dict[iti]['trials'] = np.arange(len(pA2US_sop_unconstrained_ITI_trials_dict[iti]['mean']))+1

scaled = True
fig_sop_unconstrained_val, ax_sop_unconstrained_val = ff.plot_model_sweep_timecourses(value_sop_unconstrained_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'sop_unconstrained - value',
                                                             ylabel = 'value (cue)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )

fig_sop_unconstrained_pA2US, ax_sop_unconstrained_pA2US = ff.plot_model_sweep_timecourses(pA2US_sop_unconstrained_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = scaled,
                                                             title = 'sop_unconstrained - pA2_US',
                                                             ylabel = 'pA2_US',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
#%%

"""
FIGURE 5J & EXTENDED DATA FIGURE 7G:
 - ANCCR TIMECOURSES AND ASYMPTOTE DA/ANCCR
"""

#anccr
num_trials_by_ITI = [800, 400, 88, 48, 16]
mean_asymptote_NC_by_ITI = {x: [] for x in labels_ITI}
NC_ITI_trials_dict = {x: {} for x in labels_ITI}
for idx, (iti, num_trials, NC) in enumerate(zip(labels_ITI, num_trials_by_ITI, anccr_NCcr)):
    NC_ITI_trials_dict[iti]['mean'] = np.nanmean(NC[:, :num_trials], axis = 0)
    mean_asymptote_NC_by_ITI[iti] = np.nanmean(NC[:, num_trials+trials_to_analyze_by_ITI[idx]:num_trials], axis = 1)
    NC_ITI_trials_dict[iti]['sem'] = stats.sem(NC[:, :num_trials], axis = 0, ddof = 1)
    NC_ITI_trials_dict[iti]['trials'] = np.arange(len(NC_ITI_trials_dict[iti]['mean']))+1
fig_anccr, ax_anccr = ff.plot_model_sweep_timecourses(NC_ITI_trials_dict,
                                                             colors_for_conditions = dc.colors_for_conditions,
                                                             conditions_to_plot = 'all',
                                                             scaled = True,
                                                             title = 'ANCCR',
                                                             ylabel = 'NC(c->r)',
                                                             axsize = dc.axsize_timecourse,
                                                             fig_path = fig_path_Figure5_ExtDataFig7,
                                                             save_fig = save_figs,
                                                             )
asymptote_NC = ff.plotBarsFromDict(mean_asymptote_NC_by_ITI,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='NC asymptote',
                                           data_is_nested_dict = False,
                                           data_is_regular_dict = True,
                                           save_fig = save_figs,
                                           axsize = dc.axsize_bars_3,
                                           fig_path = fig_path_Figure5_ExtDataFig7,
                                           )
itis_to_plot_DA_anccr = ['iti30', 'iti60','iti300','iti600','iti3600', ]
mean_asymptote_normDA_by_ITI = {x: [] for x in labels_ITI}
for da_label, iti_label, last_trials in zip(itis_to_plot_DA_anccr, labels_ITI, trials_to_analyze_by_ITI):
    single_iti_normDA_cue = anccr_modeldata[da_label]['normDAcuersp']
    mean_asymptote_normDA_by_ITI[iti_label] = np.nanmean(single_iti_normDA_cue[:, last_trials:], axis = 1)

asymptote_normDA = ff.plotBarsFromDict(mean_asymptote_normDA_by_ITI,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='norm DA asymptote',
                                           data_is_nested_dict = False,
                                           data_is_regular_dict = True,
                                           save_fig = save_figs,
                                           axsize = dc.axsize_bars_3,
                                           fig_path = fig_path_Figure5_ExtDataFig7,
                                           )
#stats to compare 60, 600, and 3600s (groups which have experimental dopamine data)
mean_asymptote_normDA_by_ITI_long_df = sf.convert_dict_to_long_df(mean_asymptote_normDA_by_ITI,
                                                                        label_key = 'condition',
                                                                        label_values = 'normDA_mean',
                                                                        )
kruskal_resultsh_normDA = pingouin.kruskal(data = mean_asymptote_normDA_by_ITI_long_df,
                        dv = 'normDA_mean',
                        between = 'condition',
                        )
labels_ITI_subset = ['60 s ITI', '600 s ITI', '3600 s ITI',]
normDA_mean_subset = sf.convert_dict_to_long_df(mean_asymptote_normDA_by_ITI,
                            label_key = 'condition',
                            label_values = 'normDA_mean',
                            keys_to_include = labels_ITI_subset,
                            )
kruskal_pairwise_mwu_tests_anccr_normDA = pingouin.pairwise_tests(data = normDA_mean_subset,
                                                                    dv = 'normDA_mean',
                                                                    between = 'condition',
                                                                    correction = True,
                                                                    padjust = 'bonf',
                                                                    parametric = False,
                                                                    return_desc = True)