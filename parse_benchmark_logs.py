import json
import re
import numpy as np

import matplotlib.pyplot as plt

def parse_gbench_json_log(filename):
    times_mkl = []
    times_par = []

    with open(filename) as log_file:
        log_obj = json.load(log_file)

        bw_regex = r"BM_.*/(\d+)/.*"

        for entry in log_obj["benchmarks"]:
            m = re.match(bw_regex, entry["name"])
            bw = 0
            if m:
                bw = int(m.group(1))
                if entry["name"].startswith("BM_par_pbtrf"):
                    times_par.append({"bandwidth": bw, "time": entry["real_time"]})
                elif entry["name"].startswith("BM_Lapacke"):
                    times_mkl.append({"bandwidth": bw, "time": entry["real_time"]})
                else:
                    raise Exception("Unexpected log file name")
            else:
                raise Exception("Log file has unexpected entry name")

    return (times_mkl, times_par)

def plot_time_comparison(times_mkl, times_par):
    bandwidths = []
    times = []
    mkl_t = []
    for entry in times_mkl:
        bandwidths.append(entry["bandwidth"])
        mkl_t.append(entry["time"])
    
    for entry in times_par:
        times.append(entry["time"])

    fig, ax = plt.subplots()

    ax.semilogy(bandwidths, mkl_t, marker="*")
    ax.semilogy(bandwidths, times, marker="o")
    plt.show()



if __name__ == "__main__":
    plt.style.use("bmh")
    times_mkl, times_par = parse_gbench_json_log("build/precdog_full_space_iomp_mkl.json")
    plot_time_comparison(times_mkl, times_par)










