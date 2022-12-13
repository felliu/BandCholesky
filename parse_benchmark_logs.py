import json
import re

import matplotlib.pyplot as plt
import numpy as np

def parse_flop_count(filename):
    bw_flop_dict = {}
    with open(filename) as log_file:
        for line in log_file:
            tup = list(map(int, line.split(",")))
            bw_flop_dict[tup[0]] = tup[1]
    return bw_flop_dict

#peak_gflops = 348.79
peak_gflops = 582.4
#peak_gflops = 274.8

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

median_regex = r".+/(\d+)/repeats:\d+.*median.*"
stddev_regex = r".+/(\d+)/repeats:\d+.*stddev.*"

def parse_gbench_console_log(filename):
    #This dictionary when indexed by bandwidth should give a pair of [mean, stddev] times.
    entries = {}
    with open(filename, 'r') as log_file:
        for line in log_file:
            match_median = re.match(median_regex, line)
            if match_median:
                bw = match_median.group(1)
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
    stddevs = list(map(float, stddevs))
    stddevs_top = [m + stddev for (m, stddev) in zip(real_times, stddevs)]
    stddevs_bot = [m - stddev for (m, stddev) in zip(real_times, stddevs)]

    #ax.errorbar(bandwidths, real_times, yerr=stddevs, capsize=2.0, **plot_kwargs)
    ax.plot(bandwidths, real_times, **plot_kwargs)
    ax.fill_between(bandwidths, stddevs_top, stddevs_bot, alpha=0.2)

def plot_entries_flops(ax, entries, flop_dict, **plot_kwargs):
    bandwidths = list(map(int, entries.keys()))
    bandwidths.sort()
    gflops = []
    stddevs = []
    for bw in bandwidths:
        time_sec = float(entries[str(bw)][0]) / 1000.0
        flop = flop_dict[bw] / time_sec
        gflops.append(flop / 1e9)
        stddevs.append(entries[str(bw)][1])

    gflops = list(map(float, gflops))
    stddevs = list(map(float, stddevs))
    #stddevs_top = [m + stddev for (m, stddev) in zip(flops, stddevs)]
    #stddevs_bot = [m - stddev for (m, stddev) in zip(flops, stddevs)]

    #ax.errorbar(bandwidths, real_times, yerr=stddevs, capsize=2.0, **plot_kwargs)
    ax.semilogy(bandwidths, gflops, **plot_kwargs)
    #ax.fill_between(bandwidths, stddevs_top, stddevs_bot, alpha=0.2)

def add_labels(ax, use_flops=False):
    ax.set_xlabel("bandwidth")
    if use_flops:
        ax.set_ylabel("FLOP/s")
    else:
        ax.set_ylabel("Time (ms)")
    #ax.set_title("Cholesky Performance on Kebnekaise")
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

def make_keb_logs():
    log_names = ["keb_logs/keb_seq_mkl_numactl.log",
                "keb_logs/keb_mkl_14t_numactl.log",
                "keb_logs/keb_par_mkl_fine_14t_numactl.log",
                "keb_logs/keb_par_blis_fine_numactl.log"]
    labels = ["Sequential MKL",
              "MKL (14 threads)",
              "Task Parallel + MKL (14 threads)",
              "Task Parallel + BLIS (14 threads)"]
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

def make_keb_hi_logs_single():
    log_names = ["bench_logs_keb/keb_mkl_14t_numactl_hi.log",
                 "bench_logs_keb/keb_plasma_hi_14t_numactl.log",
                 "bench_logs_keb/keb_par_fine_mkl_14t_numactl_hi.log",
                 "bench_logs_keb/keb_par_blis_fine_numactl_hi.log"]

    labels = ["MKL (14 threads)", "PLASMA (14 threads)",
              "Task Parallel + MKL (14 threads)",
              "Task Parallel + BLIS (14 threads)"]
    plot_log_entries(log_names, labels)

def make_keb_hi_logs_full():
    log_names = ["bench_logs_keb/full_node/keb_mkl_28t_hi.log",
                 "bench_logs_keb/full_node/keb_plasma_hi_28t.log",
                 "bench_logs_keb/full_node/keb_par_fine_mkl_seq_hi.log",
                 "bench_logs_keb/full_node/keb_par_blis_fine_hi.log"]

    labels = ["MKL (28 threads)", "PLASMA (28 threads)",
              "Task Parallel + MKL (28 threads)",
              "Task Parallel + BLIS (28 threads)"]
    plot_log_entries(log_names, labels)


def plot_log_entries(log_names, labels):
    flop_dict = parse_flop_count("flop_count.txt")
    markers = ["o", "x", "1", "2", "3", "."]
    fig, ax = plt.subplots()
    for i, (log_name, label) in enumerate(zip(log_names, labels)):
        entries_log_file = parse_gbench_console_log(log_name)
        plot_entries_flops(ax, entries_log_file, flop_dict, marker=markers[i], lw=0.5, label=label)

    ax.axhline(peak_gflops, linestyle="dashed", lw=1, color="gray", label="Peak performance")
    ax.annotate(str(peak_gflops) + " GFLOP/s", xy=(0.1, 0.8), xycoords="axes fraction", xytext=(0.1, 0.9), textcoords="axes fraction")

    add_labels(ax, True)
    #plt.savefig("keb_lo.pdf")
    plt.show()

if __name__ == "__main__":
    plt.style.use("bmh")
    #make_precdog_logs()
    #make_precdog_hi_logs()
    #make_keb_logs()
    #make_keb_hi_logs_full()
    make_keb_hi_logs_single()
    #parse_flop_count("flop_count.txt")










