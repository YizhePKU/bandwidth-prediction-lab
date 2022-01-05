import re
import pickle
from glob import glob
from tqdm import tqdm


def parse_txt(filename):
    """Parse a txt file into a list of records."""
    with open(filename) as file:
        data = file.read()
    # split records by emtpy lines
    records = data.split("\n\n")
    # discard the first 5 records -- they're startup diagnostics
    return records[5:]


def extract_timestamp(record):
    """Extract timestamp from a record and return it as an integer in milliseconds."""
    match = re.search(r"(\d\d):(\d\d):(\d\d)\.(\d\d\d)", record)
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))
    millisecond = int(match.group(4))
    return ((((hour * 60) + minute) * 60) + second) * 1000 + millisecond


def extract_rx_deliv(record):
    """Extract accumulated total bytes delivered from a PDCP DL Data record."""
    # This pattern matches the third integer in a table, where integers are separated
    # by whitespaces and vertical bars.
    values = re.findall(r"\|\s*\d\|\s*\d+\|\s*(\d+)\|\s*\d+\|\s*\d+\|", record)
    return max(int(x) for x in values)


def extract_5G_rsrp(record):
    """Extract serving RSRP (reference signal received power) from a Measurement Database Update record.

    The larger of the two RSRP is used as the signal strength. The value is often negative, measured in dBm."""
    values = re.findall(r"Serving RSRP Rx23\s+= ([-+.\d]+)", record)
    return max(float(x) for x in values)


def extract_4G_rsrp(record):
    """Extract serving RSRP (reference signal received power) from a Cell Meas Response record.

    The largest RSRP is used as the signal strength. The value is often negative, measured in dBm."""
    values = re.findall(r"Inst RSRP Rx\[.]\s+= ([-+.\d]+)", record)
    return max(float(x) for x in values)


def process_logfile(filepath, logtype):
    """Extract data from a log file.

    `logtype` is one of "LTE", "SA", and "NSA".

    Returns a dict with the following keys:
        rsrp, rsrp_ts: two lists that represents the time series of signal power.
        bandwidth, bandwidth_ts: two lists that represents the time series of bandwidth.
    """
    assert logtype in ("LTE", "SA", "NSA")
    records = parse_txt(filepath)
    rsrp = []
    rsrp_ts = []
    bandwidth = []
    bandwidth_ts = []
    # Data from RX Deliv is accumulative; we will take the first order difference of it.
    last_rx_deliv = None
    for record in records:
        try:
            if logtype == "LTE" and "Cell Meas Response" in record:
                rsrp.append(extract_4G_rsrp(record))
                rsrp_ts.append(extract_timestamp(record))
            elif logtype in ("SA", "NSA") and "Measurement Database Update" in record:
                rsrp.append(extract_5G_rsrp(record))
                rsrp_ts.append(extract_timestamp(record))
        except Exception as e:
            print(f"Warning: ignoring rsrp record {len(rsrp) + 1} because {e}")

        try:
            if "PDCP DL Data" in record:
                rx_deliv = extract_rx_deliv(record)
                if last_rx_deliv is None:
                    bandwidth.append(0)
                    last_rx_deliv = rx_deliv
                else:
                    bandwidth.append(rx_deliv - last_rx_deliv)
                    last_rx_deliv = rx_deliv
                bandwidth_ts.append(extract_timestamp(record))
        except Exception as e:
            print(f"Warning: ignoring bandwidth record {len(rsrp) + 1} because {e}")
    return {
        "rsrp": rsrp,
        "rsrp_ts": rsrp_ts,
        "bandwidth": bandwidth,
        "bandwidth_ts": bandwidth_ts,
    }


def process_all_data():
    """Collect all data on the disk and process them."""
    glob_patterns = lambda filetype: [
        f"/mnt/20210530/{filetype}/client*/2021-*.txt",
        f"/mnt/20210715/{filetype}/client*/2021-*.txt",
        f"/mnt/20210805/{filetype}/client/*/2021-*.txt",
        f"/mnt/20210911/{filetype}/client/*/2021-*.txt",
    ]
    result = {
        "LTE": [],
        "SA": [],
        "NSA": [],
    }
    for filetype in ("SA", "LTE", "NSA"):
        pathnames = []
        for pat in glob_patterns(filetype):
            pathnames += glob(pat)
        for pathname in tqdm(pathnames):
            result[filetype].append(process_logfile(pathname, filetype))
    return result


data = process_all_data()

with open("data.pickle", "wb") as file:
    pickle.dump(data, file=file)
