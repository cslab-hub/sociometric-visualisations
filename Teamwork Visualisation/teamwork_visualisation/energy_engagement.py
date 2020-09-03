import numpy as np
import pandas as pd

def get_series_variations(s):
    
    return s[s.shift().ne(s)]

def get_turn_durations(s):
    
    variations = get_series_variations(s)
    
    starts = variations[:-1][variations==1].index.to_series()
    
    ends = pd.Series(variations[1:][variations==0].index, index = starts.index)
    
    durations = ends - starts
    
    return durations

def get_number_turns(df):
    
    return df.apply(lambda s: get_series_variations(s)[:-1].sum())

def get_responses(df):

    ## Find starts
    ## See if person i starts while other person is talking or within some seconds following

    start_pattern = np.array([0,1])

    def find_starts(series):

        def find_start(x):

            return np.array_equal(x, start_pattern)

        return series.rolling(window=2).apply(find_start, raw=True)

    starts = df.apply(find_starts)


    ## Speech within five seconds of someone talking counts as interacting
    ## I have not found a paper that mentions what thresholds have been used
    speech_plus_five = df.rolling(window=5).max()


    ## Find interactions where person i follows person j
    responses = pd.DataFrame(index=df.columns, columns=df.columns)


    for i in df.columns:

        for j in df.columns:

            responses.loc[i,j] = (starts.loc[:,i] * speech_plus_five.loc[:,j]).sum()
            
    responses_symmetric = responses + responses.T / 2

    responses_symmetric.values[np.triu_indices(len(df.columns))] = np.nan
            
    return responses_symmetric

def get_energy(per_second_speech):
    
    return get_number_turns(per_second_speech).rename("energy") / len(per_second_speech)

def get_engagement(per_second_speech):
    
    return pd.DataFrame(get_responses(per_second_speech).stack().rename("engagement")) / len(per_second_speech)