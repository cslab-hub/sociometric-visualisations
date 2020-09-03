import pandas as pd
import numpy as np
from teamwork_visualisation.from_MIT import *

def raw_rhythm_to_aligned_df(input_file_name, log_version="2.0"):

    data = sample2data(input_file_name, log_version=log_version)

    return pd.pivot_table(data.reset_index(),index='datetime',
                              columns='member',values='signal').fillna(value=0)

def get_correlation_threshold_information(audio_data):
    
    KL_values, slots_values = [], []
    threshold_range = np.arange(0.3, 0.7, 0.1)
    
    audio_data_seconds = get_meet_sec(audio_data)
    correlations = audio_data_seconds.groupby(level=0).apply(lambda a: a.loc[:, a.any(0)].corr())
    
    for correlation_threshold in threshold_range:
        
        speak_mean, speak_std = get_spk_genuine(audio_data, correlations, correlation_threshold)
        nys_mean = get_ts_distribution(audio_data, speak_mean) ## Distributions of volume values above and below the threshold for each speaker
        KL = get_kldistance(nys_mean, plot=False, prox=0.001)[0]
        KL_values.append(np.mean(KL))
        slots_values.append(len(speak_mean))
        
    return KL_values, slots_values, threshold_range

def get_correlation_threshold_recommendation(audio_data):
    
    KL_values, slots_values, threshold_range = get_correlation_threshold_information(audio_data)
    
    return threshold_range[(np.array(KL_values) * np.array(slots_values)).argmax()]

def voice_activity_detection(audio_data, correlation_threshold):

    audio_data_seconds = get_meet_sec(audio_data)
    correlations = audio_data_seconds.groupby(level=0).apply(lambda a: a.loc[:, a.any(0)].corr())
    
    speak_mean, speak_std = get_spk_genuine(audio_data, correlations, correlation_threshold)

    ## 2nd-pass
    speak_all, seps_mean, seps_std = get_spk_all(audio_data, speak_mean, speak_std)

    ## 3rd-pass 
    return get_spk_real(audio_data, speak_all, correlations, correlation_threshold)
    
def get_per_second_speech(df):
    
    return pd.get_dummies(df["speaker"]).groupby(level=0).sum().asfreq("1S", fill_value=0)