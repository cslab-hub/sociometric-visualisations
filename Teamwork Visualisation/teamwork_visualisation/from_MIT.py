import json
import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
import scipy.stats as stats

#######################################################
## Copied from MIT and updated for newer Python/Pandas
#######################################################

### I did not want to update everything in the MIT package (to work with newer Python/Pandas), just the stuff I need
### Also I do not have access to the MIT GitHub to upload and maintain this code there
### Therefore I have copied the bits I need from the MIT package here
### Which also makes distribution easier, and gives some confidence that changes to the MIT package will not break our routines
### Note that I have made small changes to fix issues that I encountered - this code is not exactly the same as the original code copied from MIT
### The below licence covers the parts that I copied

"""
MIT License

Copyright (c) 2016, Oren Lederman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def is_meeting_metadata(json_record):
    """
    returns true if given record is a header
    :param json_record:
    :return:
    """
    if 'startTime' in json_record:
        return True
    elif "type" in json_record and json_record["type"] == "meeting started":
        return True
    else:
        return False

def meeting_log_version_from_file(file_object):
    """
        returns version number for a given file object (or None, if version cannot be identified)
        :param file_object:
        :return:
    """
    last_pos = file_object.tell() # keep current position
    first_line = file_object.readline()
    file_object.seek(last_pos) # rewind
    meeting_metadata = json.loads(first_line)  # Convert the header string into a json object
    if is_meeting_metadata(meeting_metadata):
        return meeting_log_version(meeting_metadata)
    else:
        return None

def metadata_from_file(file_object):
    """
        Returns metadata from file (or None if no metadata)
        :param file_object:
        :return:
    """
    last_pos = file_object.tell() # keep current position
    first_line = file_object.readline()
    file_object.seek(last_pos) # rewind
    meeting_metadata = json.loads(first_line)  # Convert the header string into a json object
    if is_meeting_metadata(meeting_metadata):
        return meeting_metadata
    else:
        return None

def load_audio_chunks_as_json_objects(file_object, log_version=None, ignore_errors=True):
    """
    Loads an audio chunks as jason objects
    :param file_object: a file object to read from
    :param log_version: defines the log_version if file is missing a header line
    :param ignore_errors: when set to true, skips faulty lines
    :return:
    """
    first_data_row = 0 # some file may contain meeting information/header

    raw_data = file_object.readlines()           # This is a list of strings

    if log_version == '1.0':
        first_data_row = 1
        batched_sample_data = list(map(json.loads, raw_data[first_data_row:]))  # Convert the raw sample data into a json object

    elif log_version == '2.0':
        c = 0
        batched_sample_data = []
        for row in raw_data[first_data_row:]:
            c += 1
            try:
                data = json.loads(row)
                if data['type'] == 'audio received':
                    batched_sample_data.append(data['data'])
            except Exception as e:
                s = traceback.format_exc()
                if ignore_errors:
                    print("unexpected failure in line {}, skipping it ({})".format(c, e))
                    continue
                else:
                    print("unexpected failure in line {}, {} ,{}".format(c, e, s))
                    raise

    else:
        raise Exception('Must provide log version')

    return batched_sample_data

def sample2data(input_file_path, datetime_index=True, resample=True, log_version=None, ignore_errors=True):
    """
    Loads audio data form file and converts it to audio samples.
    Note that this method is somewhat old and needs to be re-written. In particular, it currently converted timestamps
    into EST time by deducting 4 hours
    :param input_file_path:
    :param datetime_index:
    :param resample:
    :param log_version:
    :param ignore_errors:
    :return:
    """
    with open(input_file_path,'r') as input_file:
        if log_version is None:
            log_version = meeting_log_version_from_file(input_file)
        meeting_metadata = metadata_from_file(input_file)
        batched_sample_data = load_audio_chunks_as_json_objects(file_object=input_file, log_version=log_version, ignore_errors=ignore_errors)

    sample_data = []
    
    for j in range(len(batched_sample_data)):
        batch = {}
        batch.update(batched_sample_data[j]) #Create a deep copy of the jth batch of samples
        samples = batch.pop('samples')
        if log_version == '1.0':
            reference_timestamp = batch.pop('timestamp')*1000+batch.pop('timestamp_ms') #reference timestamp in milliseconds
            sampleDelay = batch.pop('sampleDelay')
        elif log_version == '2.0':
            reference_timestamp = batch.pop('timestamp')*1000 #reference timestamp in milliseconds
            sampleDelay = batch.pop('sample_period')
        numSamples = len(samples)
        #numSamples = batch.pop('numSamples')
        for i in range(numSamples):
            sample = {}
            sample.update(batch)
            sample['signal'] = samples[i]

            sample['timestamp'] = reference_timestamp + i*sampleDelay
            sample_data.append(sample)

    df_sample_data = pd.DataFrame(sample_data)
    if len(sample_data)==0:
        return None
    df_sample_data['datetime'] = pd.to_datetime(df_sample_data['timestamp'], unit='ms')
    #df_sample_data['datetime'] = df_sample_data['datetime'] - np.timedelta64(4, 'h') # note - hard coded EST time conversion
    del df_sample_data['timestamp']

    df_sample_data.sort_values('datetime')

    if(datetime_index):
        df_sample_data.set_index(pd.DatetimeIndex(df_sample_data['datetime']),inplace=True)
        #The timestamps are in UTC. Convert these to EST
        #df_sample_data.index = df_sample_data.index.tz_localize('utc').tz_convert('US/Eastern')
        df_sample_data.index.name = 'datetime'
        del df_sample_data['datetime']
        if(resample):
            grouped = df_sample_data.groupby('member')
            df_resampled = grouped.resample(rule=str(sampleDelay)+"L").mean()

    if(resample):
        # Optional: Add the meeting metadata to the dataframe
        df_resampled.metadata = meeting_metadata
        return df_resampled
    else:
        # Optional: Add the meeting metadata to the dataframe
        df_sample_data.metadata = meeting_metadata
        return df_sample_data

def get_kde_pdf(X, bandwidth=2, step=.1, num_samples=200, optimize=False):
    """
    return kde and pdf from a data sample
    """
    if len(X) ==0 :
        return [],np.array([]),[]
    if optimize:
        bandwidths = 10 ** np.linspace(-1, 1, 10)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, 
                            cv=LeaveOneOut(len(X)))
        grid.fit(X[:, None]);
        kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_params_['bandwidth']).fit(X[:,None])    
    else:
        kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(X[:,None]) 
    pdf = np.exp( kde.score_samples(np.arange(0, 100, step)[:,None]) )
    samples = kde.sample(num_samples)            
    return kde, np.array(pdf), samples

def get_meet_sec(df_meet):
    """
    return a df with a datetime index rounded to second level.
    """
    df_meet_sec = df_meet.copy()
    df_meet_sec.index = df_meet_sec.index.map(lambda x: x.replace(microsecond=0))
    return df_meet_sec

def get_seps(dt_nys, prox=0.001, step=0.1, num_samples=200, bandwidth=2):
    """
    return cut-off points for all users
    """
    seps = []
    prox = 0.01
    for idx, user in enumerate(dt_nys):
        ns, ys = dt_nys[user]
        cond_nonezero = len(ns)==0 or len(ys)==0
        kden, pns, nss = get_kde_pdf(ns, bandwidth, step, num_samples)
        kdey, pys, yss = get_kde_pdf(ys, bandwidth, step, num_samples)       
        
        pys[pys<=prox] = 0
        pns[pns<=prox] = 0

        sep = -1        
        if not cond_nonezero:
            for i in np.arange(int(100/step)-1, 0, -1):
                if pys[i-1] < pns[i-1]  and pys[i] >= pns[i]:
                    sep = i * step
                    break        
        seps.append(sep)
    seps = np.array(seps)
    seps[seps == -1] = seps[seps != -1].mean()
    return seps

def get_kldistance(dt_nys, bandwidth=2, prox=0.001, step=0.1, num_samples=200, plot=False, figsize=(12,8)):
    """
    only for 4-user situations
    calculate kl-distance of two distributions (D_t and D_s)
    """
    klds, seps = [], []
    if plot is True:
        fig, axs = plt.subplots(2,2,figsize=figsize,) 
        plt.tight_layout(h_pad=4)
    for idx, user in enumerate(dt_nys):
        ns, ys = dt_nys[user]
        cond_nonezero = len(ns) == 0 or len(ys) ==0
        kden, pns, nss = get_kde_pdf(ns, step=step, num_samples=num_samples, bandwidth=bandwidth)
        kdey, pys, yss = get_kde_pdf(ys, step=step, num_samples=num_samples, bandwidth=bandwidth)
        
        kldistance = stats.entropy(pns, pys) if not cond_nonezero else np.nan
        if not np.isinf(kldistance) and not np.isnan(kldistance):
            klds.append(kldistance)

        pys[pys<=prox] = 0
        pns[pns<=prox] = 0

        sep = -1        
        if not cond_nonezero:
            for i in np.arange(int(100/step)-1, 0, -1):
                if pys[i-1] < pns[i-1]  and pys[i] >= pns[i]:
                    sep = i * step
                    break
        seps.append(sep)
        
        if plot is True:
            ax = axs.flatten()[idx]
            sns.distplot(nss, label='Silent',  kde=False, norm_hist=True, ax=ax)
            sns.distplot(yss, label='Talking', kde=False, norm_hist=True, ax=ax)
            ax.set_title('%s kl-dist:%.2f' % (user, kldistance) )    
            ax.set_xlabel('')
            if not cond_nonezero:
                ax.axvline(x=sep)
                ax.annotate('best sep val: %.1f' % sep, xy=(sep, 0.1), xytext=(sep+5, 0.1), 
                        arrowprops= dict(facecolor='black', shrink=0.0001))
            ax.legend()
            
    seps = np.array(seps)
    seps[seps == -1] = seps[seps != -1].mean()
    return klds, seps

def get_ts_distribution(df_meet, df_spk):
    """
    get distributions for all subjects when they talk or keep silent
    """
    dt_nys = {}
    for user in df_meet.columns:
        ns = df_spk.loc[df_spk.speaker != user][user]
        ys = df_spk.loc[df_spk.speaker == user][user]
        dt_nys[user] = [ns, ys]
    return dt_nys

def get_spk_genuine(df_meet, df_cor, thre):
    """
    get genuine spk
    """
    df_meet_sec = get_meet_sec(df_meet)
    df_cor = pd.DataFrame(df_cor.fillna(1.0).gt(thre).T.all())
    df_cor.reset_index(inplace=True)
    df_cor.columns = ['datetime', 'member', 'val']
    ## Find those people whoes correlation with others are all higher than thre
    df_cor = df_cor.pivot(index='datetime', columns='member', values='val')
    df_mean_ori = df_meet_sec.groupby(df_meet_sec.index).agg(np.mean)
    df_std_ori = df_meet_sec.groupby(df_meet_sec.index).agg(np.std)
    df_mean = pd.DataFrame(df_mean_ori.T.idxmax(), columns=['speaker'])
    ## combine 'correlation' and 'volume' to locate the speaker
    df_comb = df_mean.merge(df_cor, left_index=True, right_index=True)
    ## df_comb_sel contains the speaker information 
    idx = [df_comb.loc[df_comb.index[i], u ] for i,u in enumerate(df_comb.speaker)]
    df_comb_sel = df_comb[idx][['speaker']]
    ## get speakers' mean
    df_spk_mean = df_comb_sel.merge(df_mean_ori, left_index=True, right_index=True)
    ## get their std
    df_spk_std = df_comb_sel.merge(df_std_ori, left_index=True, right_index=True)
    return df_spk_mean, df_spk_std

def get_spk_all(df_flt, df_spk_mean, df_spk_std, bandwidth=2):
    """
    get all spk
    """
    df_flt_sec = get_meet_sec(df_flt)
    gps_mean = df_flt_sec.groupby(df_flt_sec.index).mean()
    gps_std = df_flt_sec.groupby(df_flt_sec.index).std()
    speak_all = [] 
    nys_mean = get_ts_distribution(df_flt, df_spk_mean)
    nys_std = get_ts_distribution(df_flt, df_spk_std)
    seps_mean = get_seps(nys_mean, bandwidth=bandwidth)
    seps_std = get_seps(nys_std, bandwidth=bandwidth)
    for i, k in enumerate(df_flt.columns):
        volume_mean = seps_mean[i]
        volume_std = seps_std[i]
        # print '[get_speak_all] user sep_val:', k, volume_mean, volume_std
        df_std = pd.DataFrame(gps_std[gps_std[k] >= volume_std])
        df_mean = pd.DataFrame(gps_mean[gps_mean[k] >= volume_mean])
        df_mean_add = pd.DataFrame(gps_mean.loc[df_std.index])
        df_tmp = pd.concat([df_mean, df_mean_add])
        df_tmp.drop_duplicates(inplace=True)
        df_tmp['speaker'] = k
        speak_all.append(df_tmp)        
    df_speak_all = pd.concat(speak_all)
    ## miss a small fraction from tpspeak()
    # df_tmp = pd.concat([df_speak_all, df_spk_mean])
    # df_speak_all = df_tmp.drop_duplicates()
    return df_speak_all.sort_index(), seps_mean, seps_std

def get_spk_real(df_flt, df_speak_all, df_cor, thre):
    """
    get real spk
    """
    df_mean_raw = df_speak_all[df_speak_all.index.duplicated(keep = False)].sort_index()
    df_mean_raw.reset_index(inplace=True)
    vals = [df_mean_raw.loc[df_mean_raw.index[i], u ] for i, u in enumerate(df_mean_raw.speaker)]
    df_mean_sel = df_mean_raw.loc[:,['datetime', 'speaker']]
    df_mean_sel['val'] = vals
    df_mean = df_mean_sel.pivot(index='datetime', columns='speaker', values='val')
    potential_spks = df_mean.columns.tolist()
    volume_spks = [ "%s_vol" % i for i in potential_spks]
    df_mean.columns = volume_spks
    df_meet_sec = get_meet_sec(df_flt)
    ## Notie: dropna() will cause true speakers to be ignored. use fillna instead.
    df_cor = df_cor.fillna(2)
    df_comb = df_mean.merge(df_cor, left_index=True, right_index=True)
    ## I know it is hard to read... I just do not want to write for loop
    # the logic is: it returns true only when the correlation with those guys with higher volume are all smaller than the threshold
    idxs_mean = [ False if u not in potential_spks or np.isnan(df_comb.iloc[i]['%s_vol' % u]) else (df_comb.iloc[i][ df_comb.iloc[i][volume_spks][df_comb.iloc[i][volume_spks] > df_comb.iloc[i]['%s_vol' % u]].index.map(lambda x: x.rstrip('_vol'))] < thre).all() for i, u in enumerate(df_comb.index.get_level_values('member'))]
    df_ms = df_comb[idxs_mean]
    target_cols = ['member']
    target_cols.extend(volume_spks)
    df_ms = df_ms.reset_index(level='member')[target_cols]
    new_cols = ['speaker']
    new_cols.extend(potential_spks)
    df_ms.columns = new_cols
    df_ss = df_speak_all[~df_speak_all.index.duplicated(keep = False)].sort_index()
    df_speak_real = pd.concat([df_ss, df_ms]).reset_index()
    df_speak_real.set_index('datetime', inplace=True)
    return df_speak_real.sort_index()

### End of the code covered by Oren's MIT licence