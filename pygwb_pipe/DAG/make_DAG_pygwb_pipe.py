import numpy as np
from pathlib import Path
import sys
import subprocess


def sortingFunction(item):
    return int(item.stem.split("-")[2])


# Filepaths
parentdir = Path.home() / "PROJECTS/pygwb/pygwb_pipe/DAG/"
subfile = parentdir / "condor/Simulated_Data_New_Pipeline.sub"

# Args
path = Path.home() / "PROJECTS/SMDC_2021/100_day_test_pygwb/MDC_Generation_2/output/"
files_H_sorted = [f for f in path.glob("H*.*") if f.is_file()]
files_L_sorted = [f for f in path.glob("L*.*") if f.is_file()]
inipath = Path.home() / "PROJECTS/pygwb/pygwb_pipe/parameters_mock.ini"


files_H_sorted.sort(key=sortingFunction)
files_L_sorted.sort(key=sortingFunction)


n_Files = np.size(files_H_sorted)

times = np.arange(0, n_Files * 86400, 7200)

# path = "/home/arianna.renzini/PROJECTS/SMDC_2021/100_day_test_pygwb/MDC_Generation/output/"

lil_times = times[0:15]

outputdir = parentdir / "output"
logdir = outputdir / "condorLogs"

# Make directories
logdir.mkdir(parents=True, exist_ok=True)
outputdir.mkdir(parents=True, exist_ok=True)

dag = outputdir / "condor_simulated_100_day_MDC_2.dag"

with open(dag, "w") as dagfile:

    for index, time in enumerate(times):
        if time % 86400 == 0:
            index_2 = int(time / 86400)
            path_H1 = files_H_sorted[index_2]
            path_L1 = files_L_sorted[index_2]
        elif time > 99 * 86400:
            print(time)
            path_H1 = files_H_sorted[-1]
            path_L1 = files_L_sorted[-1]
            print(path_H1)
        else:
            list_path = [
                files_H_sorted[j - 1]
                for j, path_to in enumerate(files_H_sorted)
                if time < float(sortingFunction(path_to))
                and time > float(sortingFunction(files_H_sorted[j - 1]))
            ]
            list_path_L = [
                files_L_sorted[k - 1]
                for k, path_to in enumerate(files_L_sorted)
                if time < float(sortingFunction(path_to))
                and time > float(sortingFunction(files_L_sorted[k - 1]))
            ]
            path_H1 = list_path[0]
            path_L1 = list_path_L[0]

        time_of_file = sortingFunction(path_H1) + 86400
        if time + 7200 > time_of_file:
            index_file = files_H_sorted.index(path_H1)
            path_H1 = [path_H1, files_H_sorted[index_file + 1]]
            path_L1 = [path_L1, files_L_sorted[index_file + 1]]

        path_H1 = path / path_H1
        path_L1 = path / path_L1

        dagfile.write(f"JOB {index} {subfile}\n")
        dagfile.write(
            f'VARS {index} job="{index}" ARGS="--t0 {time} --tf {time+7200} '
            f"--H1 {path_H1} --L1 {path_L1} --output_path {outputdir} "
            f'--param_file {inipath}" logdir="{logdir}"\n'
        )
        dagfile.write("\n")
