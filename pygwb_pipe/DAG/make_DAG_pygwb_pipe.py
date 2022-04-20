import argparse
import numpy as np
from pathlib import Path
import sys
import subprocess


def sortingFunction(item):
    return int(item.stem.split("-")[2])


dag_parser = argparse.ArgumentParser(add_help=False)
dag_parser.add_argument(
    "--subfile", help="Submission file.", action="store", type=str
) #"condor/Simulated_Data_New_Pipeline.sub"
dag_parser.add_argument(
    "--data_path", help="Path to data files folder.", action="store", type=Path
) #"PROJECTS/SMDC_2021/100_day_test_pygwb/MDC_Generation_2/output/"
dag_parser.add_argument(
    "--parentdir", help="Starting folder.", action="store", type=Path, required=False
)
dag_parser.add_argument(
    "--param_file", help="Path to parameters.ini file.", action="store", type=str, required=False
)
dag_parser.add_argument(
    "--dag_name", help="Dag file name.", action="store", type=str, required=False
)
dag_args = dag_parser.parse_args() 

if not dag_args.parentdir:
    dag_args.parentdir = Path('./')
if not dag_args.param_file:
    dag_args.param_file = '../parameters_mock.ini'
if not dag_args.dag_name:
    dag_args.dag_name = "condor_simulated_100_day_MDC_2.dag"

# Files
files_H_sorted = [f for f in dag_args.data_path.glob("H*.*") if f.is_file()]
files_L_sorted = [f for f in dag_args.data_path.glob("L*.*") if f.is_file()]


files_H_sorted.sort(key=sortingFunction)
files_L_sorted.sort(key=sortingFunction)


n_Files = np.size(files_H_sorted)

times = np.arange(0, n_Files * 86400, 7200)

lil_times = times[0:15]

outputdir = dag_args.parentdir / "output"
logdir = outputdir / "condorLogs"

# Make directories
logdir.mkdir(parents=True, exist_ok=True)
outputdir.mkdir(parents=True, exist_ok=True)

dag = outputdir / dag_args.dag_name

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

        path_H1 = dag_args.data_path / path_H1
        path_L1 = dag_args.data_path / path_L1

        dagfile.write(f"JOB {index} {dag_args.param_file}\n")
        dagfile.write(
            f'VARS {index} job="{index}" ARGS="--t0 {time} --tf {time+7200} '
            f"--H1 {path_H1} --L1 {path_L1} --output_path {outputdir} "
            f'--param_file {dag_args.param_file}" logdir="{logdir}"\n'
        )
        dagfile.write("\n")
