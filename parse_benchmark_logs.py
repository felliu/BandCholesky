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

    ax.semilogy(bandwidths, real_times, **plot_kwargs)

def add_labels(ax):
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Cholesky Performance on Coffee Lake Workstation")
    ax.legend()

def make_precdog_logs():
    log_names = ["bench_logs/precdog_seq_mkl.log",
                "bench_logs/precdog_MKL_6t.log",
                "bench_logs/precdog_par_fine_mkl_seq.log",
                "bench_logs/precdog_par_fine_blis_seq.log"]
    labels = ["Sequential MKL",
              "MKL (6 threads)",
              "Task Parallel + MKL",
              "Task Parallel + BLIS"]
    plot_log_entries(log_names, labels)

def make_precdog_hi_logs():
    log_names = ["bench_logs/precdog_MKL_6t_hi.log",
                 "bench_logs/precdog_plasma_mkl_hi_auto_threads.log",
                 "bench_logs/precdog_par_fine_mkl_seq_hi.log",
                 "bench_logs/precdog_par_fine_blis_seq_hi.log"]

    labels = ["MKL (6 threads)", "PLASMA (6 threads)",
              "Task Parallel + MKL (6 threads)",
              "Task Parallel + BLIS (6 threads)"]
    plot_log_entries(log_names, labels)

def plot_log_entries(log_names, labels):
    markers = ["o", "x", "1", "2", "3", "."]
    fig, ax = plt.subplots()
    for i, (log_name, label) in enumerate(zip(log_names, labels)):
        entries_log_file = parse_gbench_console_log(log_name)
        plot_entries(ax, entries_log_file, marker=markers[i], lw=0.5, label=label)

    add_labels(ax)
    plt.savefig("precdog_hi_fine_log.pdf")

if __name__ == "__main__":
    plt.style.use("bmh")
    #make_precdog_logs()
    make_precdog_hi_logs()
    









