import pickle
import numpy as np
from traces import TimeSeries
from tqdm import tqdm

with open("data/raw.pickle", "rb") as file:
    data = pickle.load(file)


def sample_not_too_short(sample):
    return len(sample["rsrp"]) >= 100 and len(sample["bandwidth"]) >= 100


def discard_non_positive_values(timeseries):
    bad_ts = []
    for ts, value in timeseries:
        if value <= 0:
            bad_ts.append(ts)
    for ts in bad_ts:
        timeseries.remove(ts)


def resample(sample, inteval=50):
    """Resample bandwidth and rsrp at regular intevals.

    This converts irregular timeseries into regular ones.

    Returns:
        bw: a list of bandwidth datapoints.
        rsrp: a list of rsrp datapoints.
    """
    bw = TimeSeries(zip(sample["bandwidth_ts"], sample["bandwidth"]))
    rsrp = TimeSeries(zip(sample["rsrp_ts"], sample["rsrp"]))

    discard_non_positive_values(bw)

    # resample using only data in overlapping time
    ts_start = max(bw.first_key(), rsrp.first_key())
    ts_end = min(bw.last_key(), rsrp.last_key())
    bw1 = bw.sample(inteval, ts_start, ts_end)
    rsrp1 = rsrp.sample(inteval, ts_start, ts_end)
    assert len(bw1) == len(rsrp1)

    # discard timestamps
    bw2 = np.array([x for ts, x in bw1], dtype=np.int32)
    rsrp2 = np.array([x for ts, x in rsrp1], dtype=np.float32)
    return bw2, rsrp2


clean_data = {
    "LTE": [],
    "SA": [],
    "NSA": [],
}
for filetype in ("LTE", "SA", "NSA"):
    samples = filter(sample_not_too_short, data[filetype])
    for sample in tqdm(samples):
        clean_data[filetype].append(resample(sample))

with open("data/processed.pickle", "wb") as file:
    pickle.dump(clean_data, file=file)
