import numpy as np
import pandas as pd

def dynamic_complexity(data, window_size, difference_threshold = 0, min_val=None, max_val=None):

    if not min_val:        
        min_val = data.min()
    
    if not max_val:        
        max_val = data.max()
    
    if max_val <= min_val:        
        raise ValueError("Maximum value must be greater than minimum value")    
    
    
    ###############
    ## Fluctuation
    ###############
    
    max_fluctuation = (window_size - 1) * (max_val - min_val)
    
    def fluctuation_window(data):
        
        ## Changes from one moment to the next
        differences = data[1:] - data[:-1]
        ## Apply the threshold for treating a difference as neither increasing nor decreasing
        differences = differences * (np.abs(differences) > difference_threshold)
        differences = np.sign(differences)
    
        ## Find the indexes of the points where the gradient changes
        points_of_return = np.not_equal(differences[1:], differences[:-1])
    
        points_of_return = np.concatenate([
                [0],
                np.arange(1, len(data) - 1)[points_of_return],
                [len(data)-1]
            ])
        
        point_of_return_differences = data[points_of_return[1:]] - data[points_of_return[:-1]]
        point_of_return_sizes = points_of_return[1:] - points_of_return[:-1]
        
        return np.abs(point_of_return_differences / point_of_return_sizes).sum() / max_fluctuation
            
    
    ###############
    ## Distribution
    ###############
    
    ## How far apart points would be if there were a uniform distribution in the window
    increment = (max_val - min_val) / (window_size - 1)
            
    ## Matrix identifying how far apart is each pair of indices    
    offset = np.triu(np.flip(np.mgrid[1:window_size,1:window_size][0]) - np.flip(np.mgrid[0:window_size-1,0:window_size-1][1]))
    
    ## How many times each pair of indices appears in the summations
    repetitions = np.triu(np.mgrid[1:window_size,1:window_size][0] * np.flip(np.mgrid[1:window_size,1:window_size][1]))
    
    ## Uniform differences
    uniform_differences = offset * increment
    
    ## Normalising term
    g = increment * repetitions * offset
    
    ## Compute distribution D in a window. This can be hard to follow, but essentially the idea is to figure out how many
    ## times the various pairs of positions are used (in the matrix called 'repetitions'), since this saves on computation
    def distribution_window(data):
        
        data = np.sort(data)
    
        ## Find differences at different index locations
        differences =  np.tril(data[:, np.newaxis] - data[np.newaxis, :])[1:,:-1].T ## select [1:,:-1] to move the unneeded zeros
    
        ## Residuals
        residuals = uniform_differences - differences
        ## Heaviside function
        residuals = residuals * (residuals > 0)
    
        ## Residuals and how often they appear in the summations
        r = residuals * repetitions
    
        return 1 - r.sum()/g.sum()
    
    
    #####################
    ## Dynamic complexity
    #####################
    
    return data.rolling(window=window_size).apply(fluctuation_window, raw=True) * data.rolling(window=window_size).apply(distribution_window, raw=True)
    
    #F = data.rolling(window=window_size).apply(fluctuation_window, raw=True) ## For testing
    #D = data.rolling(window=window_size).apply(distribution_window, raw=True) ## For testing
    
    #return F * D, F, D ## For testing

def get_critical_instabilities(results, window=60, num_std=2):
    
    rolling_means = results.sum(axis=1).rolling(window=window, center=True).mean()
    rolling_stds = results.sum(axis=1).rolling(window=window, center=True).std()
    
    return results.sum(axis=1) > rolling_means + (num_std * rolling_stds)

def get_global_critical_instabilities(results, window=60, num_std=2):
    
    mean = results.sum(axis=1).mean()
    std = results.sum(axis=1).std()
    
    return results.sum(axis=1) > mean + (num_std * std)

def get_dynamic_complexity_audio(audio_data, window_size=12, **kwargs):
    
    ## Gives dynamic complexity results for a specific format of audio data. 
    
    audio_resampled = audio_data.resample("5S").mean()
    audio_resampled = audio_resampled.rename(columns=dict(zip(audio_data.columns, ["Dynamic Complexity "+str(name) for name in audio_data.columns])))
    
    return audio_resampled.apply(dynamic_complexity, window_size=window_size, **kwargs)