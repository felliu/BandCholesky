import json
import re

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

mean_regex = r".+/(\d+)/repeats:\d+.*mean.*"
stddev_regex = r".+/(\d+)/repeats:\d+.*stddev.*"

def parse_gbench_console_log(filename):
    #This dictionary when indexed by bandwidth should give a pair of [mean, stddev] times.
    entries = {}
    with open(filename, 'r') as log_file:
        for line in log_file:
            match_mean = re.match(mean_regex, line)
            if match_mean:
                bw = match_mean.group(1)
                #Each log entry should be formatted as:
                #<test_name>/<bandwidth>/repeats:<>_mean <time> ms <cpu_time> ms <iters>
                real_time = line.split()[1]
                if bw not in entries:
                    entries[bw] = [0.0, 0.0]
                
                entries[bw][0] = real_time
            match_stddev = re.match(stddev_regex, line)
            if match_stddev:
                bw = match_stddev.group(1)
                stddev = line.split()[1]
                if bw not in entries:
                    entries[bw] = [0.0, 0.0]

                entries[bw][1] = stddev
    return entries

def plot_entries(ax, entries, **plot_kwargs):
    bandwidths = list(map(int, entries.keys()))
    bandwidths.sort()
    real_times = []
    stddevs = []
    for bw in bandwidths:
        real_times.append(entries[str(bw)][0])
        stddevs.append(entries[str(bw)][1])

    real_times = list(map(float, real_times))

    ax.plot(bandwidths, real_times, **plot_kwargs)

def add_labels(ax):
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Cholesky Performance on Coffee Lake Workstation")
    ax.legend()

def make_precdog_logs():
    entries_seq_mkl = parse_gbench_console_log("bench_logs/precdog_seq_mkl.log")
    entries_par = parse_gbench_console_log("bench_logs/precdog_par_MKL_seq_3t.log")
    entries_par_mkl = parse_gbench_console_log("bench_logs/precdog_MKL_6t.log")
    entries_par_blis = parse_gbench_console_log("bench_logs/precdog_par_blis_3t.log")
    fig, ax = plt.subplots()
    plot_entries(ax, entries_seq_mkl, marker="x", lw=0.5, label="Sequential MKL")
    plot_entries(ax, entries_par_mkl, marker="o", lw=0.5, label="MKL (6 threads)")
    plot_entries(ax, entries_par, marker="1", lw=0.5, label="Task Parallel + MKL(3 threads)")
    plot_entries(ax, entries_par_blis, marker="2", lw=0.5, label="Task Parallel + BLIS (3 threads)")
    add_labels(ax)
    plt.show()

def make_precdog_hi_logs():
    entries_par_mkl = parse_gbench_console_log("bench_logs/precdog_MKL_6t_hi.log")
    entries_par_seq_MKL = parse_gbench_console_log("bench_logs/precdog_par_seq_mkl_3t_hi.log")
    entries_par_blis = parse_gbench_console_log("bench_logs/precdog_par_blis_3t2t.log")
    entries_plasma = parse_gbench_console_log("bench_logs/precdog_plasma_mkl_hi_auto_threads.log")
    fig, ax = plt.subplots()
    plot_entries(ax, entries_par_mkl, marker="o", lw=0.5, label="MKL (6 threads)")
    plot_entries(ax, entries_plasma, marker="x", lw=0.5, label="PLASMA (6 threads)")
    plot_entries(ax, entries_par_seq_MKL, marker="1", lw=0.5, label="Task Parallel + MKL (3 threads)")
    plot_entries(ax, entries_par_blis, marker="2", lw=0.5, label="Task Parallel + BLIS (6 threads total)")
    add_labels(ax)
    plt.show()

if __name__ == "__main__":
    plt.style.use("bmh")
    #make_precdog_logs()
    make_precdog_hi_logs()
    









