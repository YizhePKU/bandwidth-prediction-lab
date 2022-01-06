import pickle
import matplotlib.pyplot as plt
from traces import TimeSeries

with open('data.pickle', 'rb') as file:
    data = pickle.load(file)

def sample_not_too_short(sample):
    return len(sample['rsrp']) >= 100 and len(sample['bandwidth']) >= 100

def discard_non_positive_values(timeseries):
    bad_ts = []
    for ts, value in timeseries:
        if value <= 0:
            bad_ts.append(ts)
    for ts in bad_ts:
        timeseries.remove(ts)

def resample(sample, inteval=50):
    '''Resample bandwidth and rsrp at regular intevals.

    This converts irregular timeseries into regular ones.
    
    Returns:
        bw: a list of bandwidth datapoints.
        rsrp: a list of rsrp datapoints.
    '''
    bw = TimeSeries(zip(sample['bandwidth_ts'], sample['bandwidth']))
    rsrp = TimeSeries(zip(sample['rsrp_ts'], sample['rsrp']))

    discard_non_positive_values(bw)

    # resample using only data in overlapping time
    ts_start = max(sample['bandwidth_ts'][0], sample['rsrp_ts'][0])
    ts_end = min(sample['bandwidth_ts'][-1], sample['rsrp_ts'][-1])
    bw1 = bw.sample(inteval, ts_start, ts_end)
    rsrp1 = rsrp.sample(inteval, ts_start, ts_end)
    assert len(bw1) == len(rsrp1)

    # discard timestamps
    bw2 = [x for ts, x in bw1]
    rsrp2 = [x for ts, x in rsrp1]
    return bw2, rsrp2

sa_data = list(filter(sample_not_too_short, data['SA']))
sample = sa_data[100]
bw, rsrp = resample(sample)