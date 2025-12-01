# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

HELPER FUNCTIONS
"""
import os
import time
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.metrics import auc
from pynwb import NWBHDF5IO
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import lick_photo_functions as lpf

def convert_names_to_dandi_convention(conditions_list):
    mapping_table = str.maketrans({'%': 'percent', '_': '-'})
    dandified_list = [x.translate(mapping_table) for x in conditions_list]
    return dandified_list

def unDandify_animal_name(animal_name_dandi, directory = False, session_file_list = False):
    if not (directory or session_file_list):
        raise Exception ('both "directory" and "session_file_list" flags are False in unDandify_animal_name, check flags')
    elif directory:
        animal_without_sub = animal_name_dandi.split('sub-')[1]
        animal_without_sub = animal_without_sub.replace('percent', '%')
        animal_name = "_".join(animal_without_sub.rsplit("-", 1))
        return animal_name
    elif session_file_list:
        new_session_list = []
        for session in animal_name_dandi:
            animal_without_sub = session.split('_ses-')[0].split('sub-')[1]
            animal_without_sub = animal_without_sub.replace('percent', '%')
            animal_name = "_".join(animal_without_sub.rsplit("-", 1))
            day_num = session.split('_ses-')[1]
            old_naming_convention = animal_name + '_' + day_num +'_'
            new_session_list.append(old_naming_convention)
        return new_session_list




def get_nwb_animals_by_condition(nwb_dir_path,
                                 conditions_to_get,
                                 dopamine_only = False,
                                 dandi_naming_convention = True,
                                 ):
    if conditions_to_get == 'all':
        return [f.name
                for f
                in os.scandir(nwb_dir_path)
                if f.is_dir()
                ]
    elif isinstance(conditions_to_get, list):
        if dandi_naming_convention:
            conditions = [x.replace('%', 'percent') for x in conditions_to_get]
        else:
            conditions = conditions_to_get
    elif isinstance(conditions_to_get, str):
        if dandi_naming_convention:
            conditions = conditions_to_get.replace('percent', '%')
        conditions = [conditions_to_get]

    conditions_with_da = [x + 'D' for x in conditions]
    conditions_with_and_wo_da = conditions + conditions_with_da
    if dopamine_only:
        conditions_to_return = conditions_with_da
    else:
        conditions_to_return = conditions_with_and_wo_da
    all_animals = [f.name
                    for f
                    in os.scandir(nwb_dir_path)
                    if f.is_dir()
                    ]
    if dandi_naming_convention:
        animals_in_conditions = [x for x in all_animals if x.rsplit('-', 1)[0].split('sub-')[1] in conditions_to_return]
    else:
        animals_in_conditions = [x for x in all_animals if x.split('_')[-2] in conditions_to_return]

    return animals_in_conditions

def get_all_nwb_files_by_condition(nwb_dir_path,
                                   conditions_to_get,
                                   dopamine_only = False,
                                   dandi_naming_convention = True):
    nwb_file_info_df = pd.DataFrame()
    animals_in_conditions = get_nwb_animals_by_condition(nwb_dir_path,
                                                         conditions_to_get,
                                                         dopamine_only = dopamine_only,
                                                         dandi_naming_convention = dandi_naming_convention)
    animals_in_conditions = natsorted(animals_in_conditions)
    for animal in animals_in_conditions:
        day_files = [f.name
                        for f
                        in os.scandir(os.path.join(nwb_dir_path, animal))
                        if f.is_file()
                        ]
        day_files = natsorted(day_files)
        n_sessions = len(day_files)
        day_files_filename = day_files
        animal_filename = animal
        if dandi_naming_convention:
            animal = unDandify_animal_name(animal, directory = True)
            day_files = unDandify_animal_name(day_files, session_file_list = True)
        sex = animal.split('_')[-1][0]
        condition = animal.split('_')[0]
        if condition[-1] == 'D':
            condition = condition[:-1]
        single_animal_list = [animal] * n_sessions
        single_sex_list = [sex] * n_sessions
        single_condition_list = [condition] * n_sessions
        day_nums = [int(x.split('_')[-2][3:5]) for x in day_files]
        dopamine_recordings = [True
                               if ((x.split('_')[0][-1] == 'D'))
                               else False
                               for x
                               in day_files
                               ]
        single_animal_df = pd.DataFrame({'animal': single_animal_list,
                                         'sex': single_sex_list,
                                         'condition': single_condition_list,
                                         'day_num': day_nums,
                                         'dopamine_recording': dopamine_recordings,
                                         'nwb_file': [os.path.join(nwb_dir_path, animal_filename, x)
                                                      for x
                                                      in day_files_filename
                                                      ]
                                         })
        nwb_file_info_df = pd.concat([nwb_file_info_df, single_animal_df], ignore_index = True)
    return nwb_file_info_df

def get_event_log_from_nwb(nwb_file_path,
                           return_as_array = False,
                           ):
    with NWBHDF5IO(nwb_file_path, mode='r') as io:
        nwb_file = io.read()
        events = nwb_file.acquisition['eventLog']
        events_df = pd.DataFrame({'event_code': events.get('event_code').data,
                                'timestamp':  events.get('timestamp').data,
                                'event_flag': events.get('event_flag').data,
                                })
        if return_as_array:
            return events_df.to_numpy()
        else:
            return events_df
def get_timestamps_from_event_code(event_log,
                                   event_code,
                                   ):
    if isinstance(event_log, pd.DataFrame):
        events_ar = event_log.to_numpy()
    else:
        events_ar = event_log
    timestamps = events_ar[events_ar[:,0]==event_code][:,1]
    return timestamps

def get_event_flags_from_event_code(event_log,
                                   event_code,
                                   ):
    if isinstance(event_log, pd.DataFrame):
        events_ar = event_log.to_numpy()
    else:
        events_ar = event_log
    event_flags = events_ar[events_ar[:,0]==event_code][:,2]
    return event_flags

def get_event_flags_from_event_code_from_nwb(nwb_file_path,
                                           event_code,
                                           ):
    event_log = get_event_log_from_nwb(nwb_file_path,
                                       return_as_array = True,
                                       )
    event_flags = get_event_flags_from_event_code(event_log,
                                               event_code,
                                               )
    return event_flags

def get_timestamps_from_event_code_from_nwb(nwb_file_path,
                                           event_code,
                                           ):
    event_log = get_event_log_from_nwb(nwb_file_path,
                                       return_as_array = True,
                                       )
    timestamps = get_timestamps_from_event_code(event_log,
                                               event_code,
                                               )
    return timestamps


def unstringify_params_from_nwb(params_dict):
    for name, value in params_dict.items():
        if isinstance(value, str):
            new_val = value.strip()
            if new_val.startswith('[') and new_val.endswith(']'):
                new_val = [int(x) for x in new_val[1:-1].split(' ') if len(x)>0]
            else:
                new_val = int(new_val)
        params_dict[name] = new_val

    return params_dict

def get_params_from_nwb(nwb_file_path):

    with NWBHDF5IO(nwb_file_path, mode='r') as io:
        nwb_file = io.read()
        task_parameters = nwb_file.acquisition['task_parameters']
        params = {x:y
                for (x,y)
                in zip(task_parameters.get('param_name').data,
                       task_parameters.get('param_value').data,
                       )
                }
        params = unstringify_params_from_nwb(params)
    return params
def get_dff_from_nwb(nwb_file_path,
                     return_as_array = False,
                     ):
    with NWBHDF5IO(nwb_file_path, mode='r') as io:
        nwb_file = io.read()
        photometry_module = nwb_file.processing['photometry']
        photo_obj = photometry_module['photometry_dff']

        photo_df = pd.DataFrame({'time': photo_obj.timestamps,
                                'dff':  photo_obj.data,
                                })
        if return_as_array:
            return photo_df.to_numpy()
        else:
            return photo_df
def check_if_nwb_has_photometry(nwb_file_path):
    with NWBHDF5IO(nwb_file_path, 'r') as io:
        nwb_file = io.read()
        has_photometry = 'photometry' in nwb_file.processing
        return has_photometry
def convert_ms_to_s(time_in_ms):
    time_in_s = time_in_ms/1000
    return time_in_s

def make_trial_df_from_nwb(nwb_file_info_df,
                           total_time_window_s = 37,
                           baseline_length_s = 7,
                           return_session_df = False):
    start_timing_test = time.time()
    all_trial_data_df = pd.DataFrame()
    if return_session_df:
        all_session_data_df = pd.DataFrame()
    for ics, file_df_row in nwb_file_info_df.iterrows():
        animal = file_df_row['animal']
        sex = file_df_row['sex']
        condition = file_df_row['condition']
        day_num = file_df_row['day_num']
        params = get_params_from_nwb(file_df_row['nwb_file'])
        event_log_df = get_event_log_from_nwb(file_df_row['nwb_file'])
        matfile_end_time = get_timestamps_from_event_code(event_log_df, 0) # 0 is session end

        has_photo_data = (check_if_nwb_has_photometry(file_df_row['nwb_file']) and file_df_row['dopamine_recording'])
        if has_photo_data:
            photometry_df = get_dff_from_nwb(file_df_row['nwb_file'])
            last_recorded_photo_time = photometry_df['time'].iloc[-1]
            #for file where doric crashed and missed last cue/reward, we're going to ignore those in matlab data
            if (last_recorded_photo_time + .01)< matfile_end_time: #add 10ms buffer so that not triggered by rounding errors
                event_log_df = event_log_df[event_log_df['timestamp']<last_recorded_photo_time]
                print(f'chopping {animal} on {day_num}')

        CSplus_ar = get_timestamps_from_event_code(event_log_df, 15)
        antic_dur = convert_ms_to_s(params['CS_t_fxd'][1])
        CSplus_dur = convert_ms_to_s(params['CSdur'][0])
        CSplus_freq = params['CSfreq'][0]
        if (file_df_row['condition'] == '60s-CSminus'):
            CSminus_ar = get_timestamps_from_event_code(event_log_df, 7) #background solenoid arduino output is CS- here
            CSminus_dur = convert_ms_to_s(params['r_bgd'])
            CSminus_freq = 3
        else:
            CSminus_ar = get_timestamps_from_event_code(event_log_df, 18) #CS4 isn't used, read it to leave empty array
            CSminus_dur = 0
            CSminus_freq = 0
        if (file_df_row['condition'] == '600s-bgdmilk'):
            bgdReward_ar = get_timestamps_from_event_code(event_log_df, 7)
            bgdReward_cue_dur = 0
            bgdReward_cue_freq = 0
        else:
            bgdReward_ar = get_timestamps_from_event_code(event_log_df, 18)

        trace_dur = antic_dur - CSplus_dur
        cue_start = (-1*antic_dur)
        rewards_ar_all =  get_timestamps_from_event_code(event_log_df, 10)





        rewards_ar_CSplus = rewards_ar_all
        rewards_omission_ar = get_event_flags_from_event_code(event_log_df, 10)



        rewards_del_ar = 1 - rewards_omission_ar
        rewards_ar_CSminus = CSminus_ar + antic_dur
        trial_type_CSplus = ['reward' if x == 1 else 'omission' for x in rewards_del_ar]


        lick3s_ar = get_timestamps_from_event_code(event_log_df, 5)
        lick3s_off_ar = get_timestamps_from_event_code(event_log_df, 6)
        #sometimes a lick at the end of a session doesn't get a corresponding lick off time, use session end
        if len(lick3s_ar) > len(lick3s_off_ar):
            #append session end time as last "lick off" value
            lick3s_off_ar = np.append(lick3s_off_ar, event_log_df['timestamp'].iloc[-1])
        rewards_ar_all = np.sort(np.concatenate((rewards_ar_CSplus,rewards_ar_CSminus)))
        CStimes_ar_all = np.sort(np.concatenate((CSplus_ar,CSminus_ar)))
        temp_ITItimes = CStimes_ar_all[1:] - rewards_ar_all[:-1]
        ITItimes = np.insert(temp_ITItimes,0, CStimes_ar_all[0])

        num_CSplusTrials = len(CSplus_ar)
        num_CSminusTrials = len(CSminus_ar)

        CSplus_single_day_df = pd.DataFrame(data = {'animal': [file_df_row['animal']] * num_CSplusTrials,
                                          'sex': [file_df_row['sex']] * num_CSplusTrials,
                                          #'cohort': [file_df_row['cohort']] * num_CSplusTrials,
                                          'condition': [file_df_row['condition']] * num_CSplusTrials,
                                          'day_num': [int(file_df_row['day_num'])] * num_CSplusTrials,
                                          #'phase': [file_df_row['phase']] * num_CSplusTrials,
                                          #'day_info': [file_df_row['day_info']] * num_CSplusTrials,
                                          'cue_type': ['CS_plus'] * num_CSplusTrials,
                                          'cue_freq': [CSplus_freq]* num_CSplusTrials,
                                          'cue_on': CSplus_ar, 'cue_off': CSplus_ar + CSplus_dur,
                                          'reward_time': rewards_ar_CSplus,
                                          'trial_type': trial_type_CSplus, 'reward_del':rewards_del_ar,
                                          'reward_omit':rewards_omission_ar})

        CSplus_single_day_df['trial_num'] = [np.flatnonzero(CStimes_ar_all==x)[0]+1
                                             for x
                                             in CSplus_ar
                                             ]
        CSplus_single_day_df['preceding_ITI'] = [ITItimes[x-1]
                                                 for x
                                                 in CSplus_single_day_df['trial_num']
                                                 ]
        #extract licks around reward delivery
        lick_CSplus_raw = lpf.extractLickTimesAroundEvent(lick3s_ar, rewards_ar_CSplus,
                                                           total_time_window = total_time_window_s,
                                                           baseline_period = baseline_length_s,
                                                           )
        # take corresponding lick_off time even if its outside the "trial" timewindow we're extracting here
        lick_off_CSplus_raw = [lick3s_off_ar[np.isin(lick3s_ar, x) ]
                               for x
                               in lick_CSplus_raw
                               ]
        CSplus_single_day_df['licks_all'] = lpf.alignLickTimesToTrialEvents (lick_CSplus_raw,
                                                                             rewards_ar_CSplus
                                                                             )
        CSplus_single_day_df['licks_off_all'] = lpf.alignLickTimesToTrialEvents (lick_off_CSplus_raw,
                                                                                 rewards_ar_CSplus
                                                                                 )
        if return_session_df:
            single_session_df = pd.DataFrame(data = {'animal': [file_df_row['animal']],
                                              'sex': [file_df_row['sex']],
                                              'condition': [file_df_row['condition']],
                                              'day_num': [int(file_df_row['day_num'])],
                                               'CS_plus_on': [CSplus_ar], 'CS_plus_off': [CSplus_ar + CSplus_dur],
                                               'CS_minus_on':[CSminus_ar],
                                               'reward_time': [rewards_ar_CSplus],
                                               'reward_del':[rewards_del_ar],
                                               'bgd_reward_time': [bgdReward_ar],
                                               'lick_on': [lick3s_ar], 'lick_off': [lick3s_off_ar]})


        if has_photo_data:
            epoch_dff = lpf.extractEpoch(photometry_df['time'],
                                         photometry_df['dff'],
                                         rewards_ar_CSplus,
                                         total_time_window = total_time_window_s,
                                         baseline_period = baseline_length_s,
                                         )
            CSplus_single_day_df['epoch_dff'] = list(epoch_dff[1])
            CSplus_single_day_df['epoch_time'] = [epoch_dff[0]] * num_CSplusTrials


        CSminus_single_day_df = pd.DataFrame(data = {'animal': [file_df_row['animal']] * num_CSminusTrials,
                                                    'sex': [file_df_row['sex']] * num_CSminusTrials,
                                                    #'cohort': [file_df_row['cohort']] * num_CSminusTrials,
                                                    'condition': [file_df_row['condition']] * num_CSminusTrials,
                                                    'day_num': [int(file_df_row['day_num'])] * num_CSminusTrials,
                                                    #'phase': [file_df_row['phase']] * num_CSminusTrials,
                                                    #'day_info': [file_df_row['day_info']] * num_CSminusTrials,
                                                    'cue_type': ['CS_minus'] * num_CSminusTrials,
                                                    'cue_freq': [CSminus_freq] * num_CSminusTrials,
                                                    'cue_on': CSminus_ar, 'cue_off': CSminus_ar + CSminus_dur,
                                                    'reward_time': rewards_ar_CSminus,
                                                    'trial_type': ['distractor'] * num_CSminusTrials,
                                                    })
        CSminus_single_day_df['trial_num'] = [np.flatnonzero(CStimes_ar_all==x)[0]+1
                                              for x
                                              in CSminus_ar
                                              ]
        CSminus_single_day_df['preceding_ITI'] = [ITItimes[x-1]
                                                  for x
                                                  in CSminus_single_day_df['trial_num']
                                                  ]
        #extract licks around reward delivery
        lick_CSminus_raw = lpf.extractLickTimesAroundEvent(lick3s_ar, rewards_ar_CSminus,
                                                           total_time_window = total_time_window_s,
                                                           baseline_period = baseline_length_s
                                                           )
        # take corresponding lick_off time even if its outside the "trial" timewindow we're extracting here
        lick_off_CSminus_raw = [lick3s_off_ar[np.isin(lick3s_ar, x) ]
                                for x
                                in lick_CSminus_raw
                                ]
        CSminus_single_day_df['licks_all'] = lpf.alignLickTimesToTrialEvents (lick_CSminus_raw,
                                                                              rewards_ar_CSminus
                                                                              )
        CSminus_single_day_df['licks_off_all'] = lpf.alignLickTimesToTrialEvents (lick_off_CSminus_raw,
                                                                                  rewards_ar_CSminus
                                                                                  )
        num_bgdrewardTrials = len(bgdReward_ar)
        bgdrewards_ar_all = np.sort(np.concatenate((rewards_ar_CSplus,
                                                    bgdReward_ar)
                                                   )
                                    )
        temp_ITItimes_bgd = np.diff(bgdrewards_ar_all)
        ITItimes_bgd = np.insert(temp_ITItimes_bgd,
                                 0,
                                 bgdrewards_ar_all[0]
                                 )
        bgdReward_single_day_df = pd.DataFrame(data = {'animal': [file_df_row['animal']] * num_bgdrewardTrials,
                                                    'sex': [file_df_row['sex']] * num_bgdrewardTrials,
                                                    #'cohort': [file_df_row['cohort']] * num_bgdrewardTrials,
                                                    'condition': [file_df_row['condition']] * num_bgdrewardTrials,
                                                    'day_num': [int(file_df_row['day_num'])] * num_bgdrewardTrials,
                                                    #'phase': [file_df_row['phase']] * num_bgdrewardTrials,
                                                    #'day_info': [file_df_row['day_info']] * num_bgdrewardTrials,
                                                    'cue_type': ['none'] * num_bgdrewardTrials,
                                                    'cue_freq': [0] * num_bgdrewardTrials,
                                                    'cue_on': bgdReward_ar-1.250, 'cue_off': bgdReward_ar -1.000,
                                                    'reward_time': bgdReward_ar,
                                                    'trial_type': ['bgd_reward'] * num_bgdrewardTrials,
                                                    })
        bgdReward_single_day_df['trial_num'] = [np.flatnonzero(bgdrewards_ar_all==x)[0]+1
                                                for x
                                                in bgdReward_ar
                                                ]
        bgdReward_single_day_df['preceding_ITI'] = [ITItimes_bgd[np.flatnonzero(bgdrewards_ar_all > x)][0]
                                                    for x
                                                    in bgdReward_ar
                                                    ]
        #extract licks around reward delivery
        lick_bgdReward_raw = lpf.extractLickTimesAroundEvent(lick3s_ar, bgdReward_ar,
                                                           total_time_window = total_time_window_s,
                                                           baseline_period = baseline_length_s,
                                                           )
        # take corresponding lick_off time even if its outside the "trial" timewindow we're extracting here
        lick_off_bgdReward_raw = [lick3s_off_ar[np.isin(lick3s_ar, x)]
                                  for x
                                  in lick_bgdReward_raw
                                  ]
        bgdReward_single_day_df['licks_all'] = lpf.alignLickTimesToTrialEvents (lick_bgdReward_raw,
                                                                                bgdReward_ar
                                                                                )
        bgdReward_single_day_df['licks_off_all'] = lpf.alignLickTimesToTrialEvents (lick_off_bgdReward_raw,
                                                                                    bgdReward_ar
                                                                                    )
        bothCS_single_day_df = pd.concat([CSplus_single_day_df,
                                          CSminus_single_day_df,
                                          bgdReward_single_day_df
                                          ],
                                         axis = 0,
                                         ignore_index=True,
                                         )

        bothCS_single_day_df['cue_dur'] = [off - on
                                           for (on,
                                                off)
                                           in zip(bothCS_single_day_df['cue_on'],
                                                  bothCS_single_day_df['cue_off']
                                                  )
                                           ]
        bothCS_single_day_df['antic_dur'] = [rew - cue
                                             for (cue,
                                                  rew)
                                             in zip(bothCS_single_day_df['cue_on'],
                                                    bothCS_single_day_df['reward_time']
                                                    )
                                             ]


        #after gathering lick info calculate specific lick info each df

        #'all' here for bsln refers to a length equal to cue +trace
        bothCS_single_day_df['licks_bsln_all'] = [licktime[((licktime >= (-2*antic_dur))
                                                             & (licktime < (-1*antic_dur)))
                                                          ]
                                                  for licktime
                                                  in bothCS_single_day_df['licks_all']
                                                  ]
        bothCS_single_day_df['licks_antic_all'] = [licktime[((licktime >= (-1*antic_dur))
                                                             & (licktime < 0))
                                                            ]
                                                   for licktime
                                                   in bothCS_single_day_df['licks_all']
                                                   ]
        bothCS_single_day_df['licks_consume_5s'] = [licktime[((licktime >= 0)
                                                              & (licktime < 5.0))
                                                             ]
                                                    for licktime
                                                    in bothCS_single_day_df['licks_all']
                                                    ]
        # bothCS_single_day_df['licks_bsln_500ms'] = [licktime[np.flatnonzero(((licktime >= ((-1*antic_dur) - 500))
        #                                                               & (licktime < (-1*antic_dur)) ))
        #                                               ]
        #                                             for licktime
        #                                             in bothCS_single_day_df['licks_all']
        #                                             ]
        # bothCS_single_day_df['licks_antic_500ms'] = [licktime[((licktime >= (-1*500))
        #                                                        & (licktime < 0))
        #                                                       ]
        #                                              for licktime
        #                                              in bothCS_single_day_df['licks_all']
        #                                              ]
        # bothCS_single_day_df['licks_antic_cue_500ms'] = [licktime[((licktime >= (-1*antic_dur))
        #                                                            & (licktime < (-1*antic_dur+500)))
        #                                                           ]
        #                                                  for licktime
        #                                                  in bothCS_single_day_df['licks_all']
        #                                                  ]
        #count number of detected in licks in whole baseline, anticipatory duration, and consume
        bothCS_single_day_df['nlicks_bsln'] = [len(x)
                                               for x
                                               in bothCS_single_day_df['licks_bsln_all']
                                               ]
        bothCS_single_day_df['nlicks_antic_raw'] = [len(x)
                                                    for x
                                                    in bothCS_single_day_df['licks_antic_all']
                                                    ]
        bothCS_single_day_df['nlicks_antic_norm'] = (bothCS_single_day_df['nlicks_antic_raw']
                                                     - bothCS_single_day_df['nlicks_bsln']
                                                     )
        bothCS_single_day_df['nlicks_consume'] = [len(x)
                                                  for x
                                                  in bothCS_single_day_df['licks_consume_5s']
                                                  ]

        # bothCS_single_day_df['nlicks_bsln_500ms'] = [len(x)
        #                                              for x
        #                                              in bothCS_single_day_df['licks_bsln_500ms']
        #                                              ]
        # bothCS_single_day_df['nlicks_antic_raw_500ms'] = [len(x)
        #                                                   for x
        #                                                   in bothCS_single_day_df['licks_antic_500ms']
        #                                                   ]
        # bothCS_single_day_df['nlicks_antic_norm_500ms'] = (bothCS_single_day_df['nlicks_antic_raw_500ms']
        #                                                    - bothCS_single_day_df['nlicks_bsln_500ms']
        #                                                    )
        # bothCS_single_day_df['antic_norm_rate_change_500ms'] = [licks/(0.5)
        #                                                         for licks
        #                                                         in bothCS_single_day_df['nlicks_antic_norm_500ms']
        #                                                         ]
        # bothCS_single_day_df['nlicks_antic_raw_cue_500ms'] = [len(x)
        #                                                       for x
        #                                                       in bothCS_single_day_df['licks_antic_cue_500ms']
        #                                                       ]
        # bothCS_single_day_df['nlicks_antic_norm_cue_500ms'] = (bothCS_single_day_df['nlicks_antic_raw_cue_500ms']
        #                                                        - bothCS_single_day_df['nlicks_bsln_500ms']
        #                                                        )
        # bothCS_single_day_df['antic_norm_rate_change_cue_500ms'] = [licks/(0.5)
        #                                                             for licks
        #                                                             in bothCS_single_day_df['nlicks_antic_norm_cue_500ms']
        #                                                             ]
        bothCS_single_day_df['antic_norm_rate_change'] = [licks/(time/1)
                                                          for (licks,
                                                               time)
                                                          in zip(bothCS_single_day_df['nlicks_antic_norm'],
                                                                 bothCS_single_day_df['antic_dur']
                                                                 )
                                                          ]

        #check if lick on/lick off spans reward time, important for aligning first lick after reward
        bothCS_single_day_df['lick_spans_reward'] = [1
                                                     if ((lick[lick_off > 0].size> 0)
                                                         and (lick[lick_off > 0][0] <0)
                                                         )
                                                     else 0
                                                     for (lick,
                                                          lick_off
                                                          )
                                                     in zip(bothCS_single_day_df['licks_all'],
                                                            bothCS_single_day_df['licks_off_all']
                                                            )
                                                     ]
        bothCS_single_day_df['lick_spans_reward_dur'] = [lick_off[lick_off > 0][0] - lick[lick_off > 0][0]
                                                         if spans_reward == 1
                                                         else np.nan
                                                         for (lick,
                                                              lick_off,
                                                              spans_reward
                                                              )
                                                         in zip(bothCS_single_day_df['licks_all'],
                                                                bothCS_single_day_df['licks_off_all'],
                                                                bothCS_single_day_df['lick_spans_reward']
                                                                )
                                                         ]
        bothCS_single_day_df['mean_antic_norm_rate_change'] = bothCS_single_day_df['antic_norm_rate_change'].mean()
        #bothCS_single_day_df['mean_antic_norm_rate_change_500ms'] = bothCS_single_day_df['antic_norm_rate_change_500ms'].mean()
        #bothCS_single_day_df['matfile'] = ([file_df_row['matfile']]
        #                                     * len(bothCS_single_day_df)
        #                                     )
        if has_photo_data:
            bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'] = [time - lick[lick > 0][0]
                                                                         if ((lick[lick > 0].size> 0)
                                                                             and (spans_reward == 0)
                                                                             )
                                                                         else time
                                                                         for (time,
                                                                              lick,
                                                                              spans_reward
                                                                              )
                                                                         in zip(bothCS_single_day_df['epoch_time'],
                                                                                bothCS_single_day_df['licks_all'],
                                                                                bothCS_single_day_df['lick_spans_reward']
                                                                                )
                                                                         ]
            bothCS_single_day_df['epoch_dff_bsln_time'] = [time[((time >= (-2*antic_dur))
                                                                 & (time < cue_start))
                                                             ]
                                                           for time
                                                           in bothCS_single_day_df['epoch_time']
                                                           ]
            bothCS_single_day_df['epoch_dff_antic_time'] = [time[((time >= cue_start)
                                                                  & (time < 0))
                                                                 ]
                                                            for time
                                                            in bothCS_single_day_df['epoch_time']
                                                            ]
            bothCS_single_day_df['epoch_dff_consume_time'] = [time[((time >= 0)
                                                                    & (time < (antic_dur)))
                                                                   ]
                                                              for time
                                                              in bothCS_single_day_df['epoch_time']
                                                              ]
            #peak and auc measurements for window corresponding to entire anticipatory period len
            #peak
            bothCS_single_day_df['epoch_dff_bsln_mean'] = [np.mean(dff[((time >= (-2*antic_dur))
                                                                        & (time < cue_start))
                                                                       ]
                                                                   )
                                                           for (dff,
                                                                time)
                                                           in zip(bothCS_single_day_df['epoch_dff'],
                                                                  bothCS_single_day_df['epoch_time'])
                                                           ]
            bothCS_single_day_df['epoch_dff_peak_antic_raw'] = [np.max(dff[((time >= cue_start)
                                                                            & (time < 0))
                                                                           ]
                                                                       )
                                                                for (dff,
                                                                     time)
                                                                in zip(bothCS_single_day_df['epoch_dff'],
                                                                       bothCS_single_day_df['epoch_time'])
                                                                ]
            bothCS_single_day_df['epoch_dff_peak_consume_raw'] = [np.max(dff[((time >= 0)
                                                                              & (time < (antic_dur)))
                                                                             ]
                                                                         )
                                                                  for (dff,
                                                                       time)
                                                                  in zip(bothCS_single_day_df['epoch_dff'],
                                                                         bothCS_single_day_df['epoch_time'])
                                                                  ]
            bothCS_single_day_df['epoch_dff_peak_antic_norm'] = (bothCS_single_day_df['epoch_dff_peak_antic_raw']
                                                                 - bothCS_single_day_df['epoch_dff_bsln_mean']
                                                                 )
            bothCS_single_day_df['epoch_dff_peak_consume_norm'] = (bothCS_single_day_df['epoch_dff_peak_consume_raw']
                                                                   - bothCS_single_day_df['epoch_dff_bsln_mean']
                                                                   )
            bothCS_single_day_df['epoch_dff_peak_consume_raw_lickaligned'] = [np.max(dff[((time >= 0)
                                                                                          & (time < (antic_dur)))
                                                                                         ]
                                                                                     )
                                                                              for (dff,
                                                                                   time)
                                                                              in zip(bothCS_single_day_df['epoch_dff'],
                                                                                     bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'])
                                                                              ]
            bothCS_single_day_df['epoch_dff_peak_consume_norm_lickaligned'] = (bothCS_single_day_df['epoch_dff_peak_consume_raw_lickaligned']
                                                                               - bothCS_single_day_df['epoch_dff_bsln_mean']
                                                                               )
            # bothCS_single_day_df['epoch_dff_dip_consume_raw_lickaligned'] = [np.min(dff[((time >= 0)
            #                                                                              & (time < (antic_dur)))
            #                                                                             ]
            #                                                                         )
            #                                                                  for (dff,
            #                                                                       time)
            #                                                                  in zip(bothCS_single_day_df['epoch_dff'],
            #                                                                         bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'])
            #                                                                  ]
            # bothCS_single_day_df['epoch_dff_dip_consume_norm_lickaligned'] = (bothCS_single_day_df['epoch_dff_dip_consume_raw_lickaligned']
            #                                                                   - bothCS_single_day_df['epoch_dff_bsln_mean']
            #                                                                   )
            #AUC
            bothCS_single_day_df['epoch_dff_bsln_auc'] = [auc(time[((time >= (-2*antic_dur))
                                                                    & (time < cue_start))
                                                                   ],
                                                              dff[((time >= (-2*antic_dur))
                                                                   & (time < cue_start))
                                                                  ]
                                                              )
                                                          for (dff,
                                                               time)
                                                          in zip(bothCS_single_day_df['epoch_dff'],
                                                                 bothCS_single_day_df['epoch_time'])
                                                          ]
            bothCS_single_day_df['epoch_dff_auc_antic_raw'] = [auc(time[((time >= cue_start)
                                                                         & (time < 0))
                                                                        ],
                                                                   dff[((time >= cue_start)
                                                                        & (time < 0))
                                                                       ]
                                                                   )
                                                               for (dff,
                                                                    time)
                                                               in zip(bothCS_single_day_df['epoch_dff'],
                                                                      bothCS_single_day_df['epoch_time'])
                                                               ]
            # bothCS_single_day_df['epoch_dff_auc_consume_raw'] = [auc(time[((time >= 0)
            #                                                                & (time < (antic_dur)))
            #                                                               ],
            #                                                          dff[((time >= 0)
            #                                                               & (time < (antic_dur)))
            #                                                              ]
            #                                                          )
            #                                                      for (dff,
            #                                                           time)
            #                                                      in zip(bothCS_single_day_df['epoch_dff'],
            #                                                             bothCS_single_day_df['epoch_time'])
            #                                                      ]
            # bothCS_single_day_df['epoch_dff_auc_antic_norm'] = (bothCS_single_day_df['epoch_dff_auc_antic_raw']
            #                                                     - bothCS_single_day_df['epoch_dff_bsln_auc']
            #                                                     )
            # bothCS_single_day_df['epoch_dff_auc_consume_norm'] =  (bothCS_single_day_df['epoch_dff_auc_consume_raw']
            #                                                        - bothCS_single_day_df['epoch_dff_bsln_auc']
            #                                                        )
            # bothCS_single_day_df['epoch_dff_auc_consume_raw_lickaligned'] = [auc(time[(time >= 0)
            #                                                                           & (time < (antic_dur))
            #                                                                           ],
            #                                                                      dff[(time >= 0)
            #                                                                          & (time < (antic_dur))
            #                                                                          ]
            #                                                                      )
            #                                                                  if (isinstance(dff, np.ndarray)
            #                                                                      & (isinstance(time, np.ndarray)))
            #                                                                  else np.nan
            #                                                                  for (dff,
            #                                                                       time)
            #                                                                  in zip(bothCS_single_day_df['epoch_dff'],
            #                                                                         bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'])
            #                                                                  ]

            # bothCS_single_day_df['epoch_dff_auc_consume_norm_lickaligned'] = (bothCS_single_day_df['epoch_dff_auc_consume_raw_lickaligned']
            #                                                                   - bothCS_single_day_df['epoch_dff_bsln_auc']
            #                                                                  )
            #peak and auc measurements for 500ms window immediately following cue/reward
            #peak
            measure_wind_ms = 0.500

            bothCS_single_day_df['epoch_dff_bsln_mean_500ms'] = [np.mean(dff[((time >= (cue_start - measure_wind_ms))
                                                                              & (time < cue_start))
                                                                             ]
                                                                         )
                                                                 if isinstance(dff, np.ndarray)
                                                                 else np.nan
                                                                 for (dff,
                                                                      time,
                                                                      cue_dur)
                                                                 in zip(bothCS_single_day_df['epoch_dff'],
                                                                        bothCS_single_day_df['epoch_time'],
                                                                        bothCS_single_day_df['cue_dur'])
                                                                 ]
            bothCS_single_day_df['epoch_dff_peak_antic_raw_500ms'] = [np.max(dff[((time >= cue_start)
                                                                                  & (time < (cue_start + measure_wind_ms)))
                                                                                 ]
                                                                             )
                                                                      if isinstance(dff, np.ndarray)
                                                                      else np.nan
                                                                      for (dff,
                                                                           time,
                                                                           cue_dur)
                                                                      in zip(bothCS_single_day_df['epoch_dff'],
                                                                             bothCS_single_day_df['epoch_time'],
                                                                             bothCS_single_day_df['cue_dur'])
                                                                      ]
            bothCS_single_day_df['epoch_dff_peak_consume_raw_500ms'] = [np.max(dff[((time >= 0)
                                                                                    & (time < measure_wind_ms))
                                                                                   ])
                                                                        if isinstance(dff, np.ndarray)
                                                                        else np.nan
                                                                        for (dff,
                                                                             time,
                                                                             cue_dur)
                                                                        in zip(bothCS_single_day_df['epoch_dff'],
                                                                               bothCS_single_day_df['epoch_time'],
                                                                               bothCS_single_day_df['cue_dur'])
                                                                        ]
            bothCS_single_day_df['epoch_dff_peak_antic_norm_500ms'] = (bothCS_single_day_df['epoch_dff_peak_antic_raw_500ms']
                                                                       - bothCS_single_day_df['epoch_dff_bsln_mean_500ms']
                                                                       )
            bothCS_single_day_df['epoch_dff_peak_consume_norm_500ms'] = (bothCS_single_day_df['epoch_dff_peak_consume_raw_500ms']
                                                                         - bothCS_single_day_df['epoch_dff_bsln_mean_500ms']
                                                                        )
            bothCS_single_day_df['epoch_dff_peak_consume_raw_lickaligned_500ms'] = [np.max(dff[((time >= 0)
                                                                                                & (time < measure_wind_ms))
                                                                                               ]
                                                                                           )
                                                                                    if isinstance(dff, np.ndarray)
                                                                                    else np.nan
                                                                                    for (dff,
                                                                                         time,
                                                                                         cue_dur)
                                                                                    in zip(bothCS_single_day_df['epoch_dff'],
                                                                                           bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'],
                                                                                           bothCS_single_day_df['cue_dur'])
                                                                                    ]

            bothCS_single_day_df['epoch_dff_peak_consume_norm_lickaligned_500ms'] = (bothCS_single_day_df['epoch_dff_peak_consume_raw_lickaligned_500ms']
                                                                                     - bothCS_single_day_df['epoch_dff_bsln_mean_500ms']
                                                                                     )
            #AUC DA using 500ms
            bothCS_single_day_df['epoch_dff_bsln_auc_500ms'] = [auc(time[((time >= (cue_start
                                                                                    -measure_wind_ms))
                                                                          & (time < cue_start))
                                                                         ],
                                                                    dff[((time >= (cue_start
                                                                                   - measure_wind_ms))
                                                                         & (time < cue_start))
                                                                        ]
                                                                    )
                                                                if isinstance(dff, np.ndarray)
                                                                else np.nan
                                                                for (dff,
                                                                     time,
                                                                     cue_dur)
                                                                in zip(bothCS_single_day_df['epoch_dff'],
                                                                       bothCS_single_day_df['epoch_time'],
                                                                       bothCS_single_day_df['cue_dur'])
                                                                ]
            bothCS_single_day_df['epoch_dff_auc_antic_raw_500ms'] = [auc(time[((time >= cue_start)
                                                                              & (time < (cue_start
                                                                                         + measure_wind_ms)))
                                                                              ],
                                                                         dff[((time >= cue_start)
                                                                             & (time < (cue_start
                                                                                        + measure_wind_ms)))
                                                                             ]
                                                                         )
                                                                     if isinstance(dff, np.ndarray)
                                                                     else np.nan
                                                                     for (dff,
                                                                          time,
                                                                          cue_dur)
                                                                     in zip(bothCS_single_day_df['epoch_dff'],
                                                                            bothCS_single_day_df['epoch_time'],
                                                                            bothCS_single_day_df['cue_dur'])
                                                                     ]
            bothCS_single_day_df['epoch_dff_auc_consume_raw_500ms'] = [auc(time[((time >= 0)
                                                                                & (time < measure_wind_ms))
                                                                                ],
                                                                           dff[((time >= 0)
                                                                               & (time < measure_wind_ms))
                                                                               ]
                                                                           )
                                                                       if isinstance(dff, np.ndarray)
                                                                       else np.nan
                                                                       for (dff,
                                                                            time,
                                                                            cue_dur)
                                                                       in zip(bothCS_single_day_df['epoch_dff'],
                                                                              bothCS_single_day_df['epoch_time'],
                                                                              bothCS_single_day_df['cue_dur'])
                                                                       ]
            bothCS_single_day_df['epoch_dff_auc_antic_norm_500ms'] = (bothCS_single_day_df['epoch_dff_auc_antic_raw_500ms']
                                                                      - bothCS_single_day_df['epoch_dff_bsln_auc_500ms']
                                                                      )
            bothCS_single_day_df['epoch_dff_auc_consume_norm_500ms'] =  (bothCS_single_day_df['epoch_dff_auc_consume_raw_500ms']
                                                                         - bothCS_single_day_df['epoch_dff_bsln_auc_500ms']
                                                                         )
            bothCS_single_day_df['epoch_dff_auc_consume_raw_lickaligned_500ms'] =  [auc(time[((time >= 0)
                                                                                              & (time < measure_wind_ms))
                                                                                             ],
                                                                                        dff[((time >= 0) & (time < measure_wind_ms))]
                                                                                        )
                                                                                    if (isinstance(dff, np.ndarray)
                                                                                        & (isinstance(time, np.ndarray)))
                                                                                    else np.nan
                                                                                    for (dff,
                                                                                         time,
                                                                                         cue_dur)
                                                                                    in zip(bothCS_single_day_df['epoch_dff'],
                                                                                           bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'],
                                                                                           bothCS_single_day_df['cue_dur'])
                                                                                    ]
            bothCS_single_day_df['epoch_dff_auc_consume_norm_lickaligned_500ms'] = (bothCS_single_day_df['epoch_dff_auc_consume_raw_lickaligned_500ms']
                                                                                    - bothCS_single_day_df['epoch_dff_bsln_auc_500ms']
                                                                                    )


            consume_measurement_wind_list = [1, 1.5, 2, 2.5, 3]
            consume_measurement_wind_list = [2]

            for consume_measurement_wind in consume_measurement_wind_list:
                measure_wind_ms = int(consume_measurement_wind*1)
                wind_str = str(consume_measurement_wind).replace('.', '')

                bothCS_single_day_df[f'epoch_dff_bsln_mean_{wind_str}s'] = [np.mean(dff[((time >= (cue_start - measure_wind_ms))
                                                                                  & (time < cue_start))
                                                                                 ]
                                                                             )
                                                                     if isinstance(dff, np.ndarray)
                                                                     else np.nan
                                                                     for (dff,
                                                                          time,
                                                                          cue_dur)
                                                                     in zip(bothCS_single_day_df['epoch_dff'],
                                                                            bothCS_single_day_df['epoch_time'],
                                                                            bothCS_single_day_df['cue_dur'])
                                                                     ]

                bothCS_single_day_df[f'epoch_dff_peak_consume_raw_{wind_str}s'] = [np.max(dff[((time >= 0)
                                                                                        & (time < measure_wind_ms))
                                                                                       ])
                                                                            if isinstance(dff, np.ndarray)
                                                                            else np.nan
                                                                            for (dff,
                                                                                 time,
                                                                                 cue_dur)
                                                                            in zip(bothCS_single_day_df['epoch_dff'],
                                                                                   bothCS_single_day_df['epoch_time'],
                                                                                   bothCS_single_day_df['cue_dur'])
                                                                            ]
                bothCS_single_day_df[f'epoch_dff_peak_consume_norm_{wind_str}s'] = (bothCS_single_day_df[f'epoch_dff_peak_consume_raw_{wind_str}s']
                                                                             - bothCS_single_day_df[f'epoch_dff_bsln_mean_{wind_str}s']
                                                                            )
                bothCS_single_day_df[f'epoch_dff_peak_consume_raw_lickaligned_{wind_str}s'] = [np.max(dff[((time >= 0)
                                                                                                    & (time < measure_wind_ms))
                                                                                                   ]
                                                                                               )
                                                                                        if isinstance(dff, np.ndarray)
                                                                                        else np.nan
                                                                                        for (dff,
                                                                                             time,
                                                                                             cue_dur)
                                                                                        in zip(bothCS_single_day_df['epoch_dff'],
                                                                                               bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'],
                                                                                               bothCS_single_day_df['cue_dur'])
                                                                                        ]

                bothCS_single_day_df[f'epoch_dff_peak_consume_norm_lickaligned_{wind_str}s'] = (bothCS_single_day_df[f'epoch_dff_peak_consume_raw_lickaligned_{wind_str}s']
                                                                                         - bothCS_single_day_df[f'epoch_dff_bsln_mean_{wind_str}s']
                                                                                         )
                #AUC DA using 500ms
                bothCS_single_day_df[f'epoch_dff_bsln_auc_{wind_str}s'] = [auc(time[((time >= (cue_start
                                                                                        -measure_wind_ms))
                                                                              & (time < cue_start))
                                                                             ],
                                                                        dff[((time >= (cue_start
                                                                                       - measure_wind_ms))
                                                                             & (time < cue_start))
                                                                            ]
                                                                        )
                                                                    if isinstance(dff, np.ndarray)
                                                                    else np.nan
                                                                    for (dff,
                                                                         time,
                                                                         cue_dur)
                                                                    in zip(bothCS_single_day_df['epoch_dff'],
                                                                           bothCS_single_day_df['epoch_time'],
                                                                           bothCS_single_day_df['cue_dur'])
                                                                    ]

                bothCS_single_day_df[f'epoch_dff_auc_consume_raw_{wind_str}s'] = [auc(time[((time >= 0)
                                                                                    & (time < measure_wind_ms))
                                                                                    ],
                                                                               dff[((time >= 0)
                                                                                   & (time < measure_wind_ms))
                                                                                   ]
                                                                               )
                                                                           if isinstance(dff, np.ndarray)
                                                                           else np.nan
                                                                           for (dff,
                                                                                time,
                                                                                cue_dur)
                                                                           in zip(bothCS_single_day_df['epoch_dff'],
                                                                                  bothCS_single_day_df['epoch_time'],
                                                                                  bothCS_single_day_df['cue_dur'])
                                                                           ]

                bothCS_single_day_df[f'epoch_dff_auc_consume_norm_{wind_str}s'] =  (bothCS_single_day_df[f'epoch_dff_auc_consume_raw_{wind_str}s']
                                                                             - bothCS_single_day_df[f'epoch_dff_bsln_auc_{wind_str}s']
                                                                             )
                bothCS_single_day_df[f'epoch_dff_auc_consume_raw_lickaligned_{wind_str}s'] =  [auc(time[((time >= 0)
                                                                                                  & (time < measure_wind_ms))
                                                                                                 ],
                                                                                            dff[((time >= 0) & (time < measure_wind_ms))]
                                                                                            )
                                                                                        if (isinstance(dff, np.ndarray)
                                                                                            & (isinstance(time, np.ndarray)))
                                                                                        else np.nan
                                                                                        for (dff,
                                                                                             time,
                                                                                             cue_dur)
                                                                                        in zip(bothCS_single_day_df['epoch_dff'],
                                                                                               bothCS_single_day_df['epoch_dff_rewardlick_aligned_time'],
                                                                                               bothCS_single_day_df['cue_dur'])
                                                                                        ]
                bothCS_single_day_df[f'epoch_dff_auc_consume_norm_lickaligned_{wind_str}s'] = (bothCS_single_day_df[f'epoch_dff_auc_consume_raw_lickaligned_{wind_str}s']
                                                                                        - bothCS_single_day_df[f'epoch_dff_bsln_auc_{wind_str}s']
                                                                                        )
        all_trial_data_df = pd.concat([all_trial_data_df,
                                       bothCS_single_day_df
                                       ],
                                      axis = 0,
                                      ignore_index=True)
        if return_session_df:
            all_session_data_df = pd.concat([all_session_data_df,
                                           single_session_df
                                           ],
                                          axis = 0,
                                          ignore_index=True)
    end_timing_test = time.time()
    print(end_timing_test - start_timing_test)
    df= all_trial_data_df.copy()
    df = lpf.cumLickTrialCount(df, grouping_var = ['animal', 'cue_type'])
    df = lpf.dropInitialNoLickTrials(df, trials_to_drop = 'consume',
                                         conditions_to_exclude= ['60s-50%',
                                                                 '60s-10%',
                                                                ])


    df = lpf.cumLickTrialCount(df, grouping_var = ['animal', 'cue_type'])
    if return_session_df:
        return all_trial_data_df, df, all_session_data_df
    return all_trial_data_df, df


def get_behavior_trials_CSplus_df(df, full3600 = False):
    if full3600:
        trials_3600 = 16
    else:
        trials_3600 = 8
    df_behavior_trials_CSplus = df[((((df['condition']=='600s')
                                          & (df['cue_trial_num'] <=40))
                                       | ((df['condition']=='60s')
                                          & (df['cue_trial_num'] <=400))
                                       | ((df['condition'] == '300s')
                                          & (df['cue_trial_num'] <=80))
                                       | ((df['condition'] == '30s')
                                          & (df['cue_trial_num'] <=800))
                                       | ((df['condition'] == '60s-few-ctxt')
                                          & (df['cue_trial_num'] <=40))
                                       | ((df['condition'] == '60s-few')
                                          & (df['cue_trial_num'] <=40))
                                       | ((df['condition'] == '60s-CSminus')
                                          & (df['cue_trial_num'] <=40))
                                       | ((df['condition'] == '600s-bgdmilk')
                                          & (df['cue_trial_num'] <=40))
                                       | ((df['condition'] == '60s-50%')
                                          & (df['cue_trial_num'] <=600))
                                       | ((df['condition'] == '60s-10%')
                                          & (df['cue_trial_num'] <=1600))
                                       | ((df['condition'] == '3600s')
                                          & (df['cue_trial_num'] <= trials_3600)))
                                       & (df['cue_type'] == 'CS_plus'))].copy()

    return df_behavior_trials_CSplus


def get_behavior_days_CSplus_df(df):

    df_behavior_days_CSplus = df[(((((df['condition']=='600s')
                                   | (df['condition']=='60s')
                                   | (df['condition'] == '300s')
                                   | (df['condition'] == '30s')
                                   | (df['condition'] == '60s-few-ctxt')
                                   | (df['condition'] == '60s-few')
                                   | (df['condition'] == '60s-CSminus')
                                   | (df['condition'] == '600s-bgdmilk'))
                                      & (df['day_num'] <=8))
                                   | ((df['condition'] == '60s-50%')
                                      & (df['day_num'] <=12))
                                   | ((df['condition'] == '3600s')
                                      & (df['day_num'] <=8))
                                   | ((df['condition'] == '60s-10%')
                                      & (df['day_num'] <=32)))
                                   & (df['cue_type'] == 'CS_plus')) ].copy()

    return df_behavior_days_CSplus

def get_behavior_trials_CSplus_learners_df(df, full3600 = False, conditions_to_exclude = ['60s-10%']):

    df_behavior_days_CSplus = get_behavior_days_CSplus_df(df)
    nonlearners_list = lpf.get_nonlearners(df_behavior_days_CSplus,
                                           conditions_to_exclude = conditions_to_exclude )
    print(nonlearners_list)
    df_behavior_trials_CSplus = get_behavior_trials_CSplus_df(df, full3600 = full3600)
    df_behavior_trials_CSplus_learners = df_behavior_trials_CSplus[(~df_behavior_trials_CSplus['animal'].isin(nonlearners_list))].copy()

    return df_behavior_trials_CSplus_learners
def get_behavior_trials_CSplus_nonlearners_df(df, full3600 = False):

    df_behavior_days_CSplus = get_behavior_days_CSplus_df(df)
    nonlearners_list = lpf.get_nonlearners(df_behavior_days_CSplus,
                                           learning_threshold = 0.5,
                                           day_above_threshold = 2,
                                           )
    df_behavior_trials_CSplus = get_behavior_trials_CSplus_df(df, full3600 = full3600)
    df_behavior_trials_CSplus_nonlearners = df_behavior_trials_CSplus[(df_behavior_trials_CSplus['animal'].isin(nonlearners_list))].copy()

    return df_behavior_trials_CSplus_nonlearners

def subset_dopamine_animals(df):
    dopamine_only = df[df['animal'].str.split('_').str[0].str[-1] == 'D'].copy()
    return dopamine_only

def get_CSminus_renamed_df(df):
    df_CSminus_days = df[((df['condition'] == '60s-CSminus')
                           & (df['day_num'] <=8))].copy()
    df_CSminus_days = lpf.cumLickTrialCount(df_CSminus_days,
                                        grouping_var = ['animal',
                                                        'cue_type']
                                        )
    rename_dict_CS_plus = {'60s-CSminus': 'CS_plus'}
    rename_dict_CS_minus = {'60s-CSminus': 'CS_minus'}
    df_CSminus_renamed_CSplus = df_CSminus_days[df_CSminus_days['cue_type'] == 'CS_plus'].replace({'condition': rename_dict_CS_plus},
                                                                      inplace = False)
    df_CSminus_renamed_CSminus = df_CSminus_days[df_CSminus_days['cue_type'] == 'CS_minus'].replace({'condition': rename_dict_CS_minus},
                                                                       inplace = False)
    df_CSminus_new = pd.concat([df_CSminus_renamed_CSplus, df_CSminus_renamed_CSminus], axis = 0, ignore_index = True)

    return df_CSminus_new


def get_all_trials_before_40th_CSplus(df):
    df_CSminus_days = df[((df['condition'] == '60s-CSminus')
                           & (df['day_num'] <=8))].copy()
    sorted_df_CSminus_days = df_CSminus_days.sort_values(by = ['animal', 'day_num', 'trial_num'])
    sorted_df_CSminus_days['cum_total_trial_count'] = sorted_df_CSminus_days.groupby('animal').cumcount()+1
    sorted_df_CSminus_days['trial_count_CS+'] = [x
                                                 if y =='CS_plus'
                                                 else np.nan
                                                 for (x,
                                                      y)
                                                 in zip (sorted_df_CSminus_days['cue_trial_num'],
                                                         sorted_df_CSminus_days['cue_type'])
                                                 ]
    excluded_sorted_df_CSminus_days = (sorted_df_CSminus_days
                                        .groupby('animal')
                                        .apply(lambda g: g[g['trial_count_CS+']
                                                           .eq(40).cumsum()
                                                           .eq(0)],
                                               ).reset_index(drop=True)
                                        )
    return excluded_sorted_df_CSminus_days

def get_renamed_learners_nonlearners_df(trial_df_learners,
                                        trial_df_nonlearners,
                                        condition_to_replace = '60s-50%',
                                        ):
    trial_df_learners = lpf.cumLickTrialCount(trial_df_learners, grouping_var = ['animal', 'cue_type'])
    trial_df_nonlearners = lpf.cumLickTrialCount(trial_df_nonlearners, grouping_var = ['animal', 'cue_type'])

    rename_dict_learners = {condition_to_replace: 'learners'}
    rename_dict_nonlearners = {condition_to_replace: 'non-learners'}
    new_learners = trial_df_learners.replace({'condition': rename_dict_learners})
    new_nonlearners = trial_df_nonlearners.replace({'condition': rename_dict_nonlearners})
    new_omissions_df = pd.concat([new_learners, new_nonlearners], axis = 0, ignore_index = True)
    return new_omissions_df