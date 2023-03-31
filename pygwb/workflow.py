# -*- coding: utf-8 -*-
# Copyright (C) California Institute of Technology 2023
#
# This file is part of pygwb.

"""Core workflow functions"""

import os
import shutil
from collections.abc import Iterable
from getpass import getuser
import numpy as np

from pycondor import Dagman as pyDagman
from pycondor import Job as pyJob
from pycondor.job import JobArg

ACCOUNTING_GROUP_USER = os.getenv(
    '_CONDOR_ACCOUNTING_USER',
    getuser(),
)

def makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def config_override(config, overrides):
    if not overrides:
        return config 
    for ov in overrides:
        split_ov = ov.split(':')
        try:
            ov_sec, ov_key, ov_val = \
                split_ov[0], split_ov[1], ':'.join(split_ov[2:])
        except:
            raise ValueError(f"Overrides must be in the form 'section:key:value', you used '{ov}'")
        if ov_sec not in config.sections():
            config[ov_sec] = {}    
        config[ov_sec][ov_key] = ov_val
    return config

def config_write(config, output_loc):
    with open(output_loc, 'w') as configfile:
        config.write(configfile)


class Job(pyJob):
    """
    Wrapper class of pycondor.Job
    """

    def __init__(self, name, executable, output_arg=None, 
                 output_file=None, accounting_group=None, 
                 extra_lines=[], arguments=None,retry=None,
                 final_result=False, input_file=None,
                 request_disk='2048MB', request_memory='512MB',
                 required_job=False,
                 **kwargs):

        exec_path = shutil.which(executable)
        if exec_path is None:
            raise TypeError('execuatble must be installed in environment'+
                       ' or given as an absolute path')

        if accounting_group is not None:
            condorcmds = [
                f"accounting_group = {accounting_group}",
                f"accounting_group_user = {ACCOUNTING_GROUP_USER}",
                ]
            condorcmds.extend(extra_lines)
        else:
            raise TypeError('accounting group must be supplied')

        output_args = []
        self.output_file = None
        # attrs for cache system
        self.output_exits = False
        self.required_job = False
            
        # output args
        if output_file is not None:
            if output_arg:
                if isinstance(output_arg, str):
                    output_args.append(output_arg)
                else:
                    raise TypeError('output_arg must be a string')
            if isinstance(output_file, str):
                self.output_file = [output_file]
                output_args.append(output_file)
            elif isinstance(output_file, Iterable):
                self.output_file = output_file
                for arg in output_file:
                    output_args.append(arg) 
            else:
                raise TypeError('output_files must be a string or an iterable')

        #input_args
        self.input_file = None
        if input_file is not None:
            if isinstance(input_file, str):
                self.input_file = [input_file]
            elif isinstance(input_file, Iterable):
                self.input_file = input_file

        # final result? 
        self.final_result = final_result
                
        arguments = ' '.join(list([arguments])+list(output_args))

        # release if requested memory is too small
        condorcmds.extend(["request_memory = ifthenelse(isUndefined(MemoryUsage),2000,3*MemoryUsage)",
                           "periodic_release = (HoldReasonCode == 26) && (JobStatus == 5)"])

        if retry == None:
            retry = 3
            
        super(Job,self).__init__(name, exec_path, arguments=arguments,
                                 extra_lines=condorcmds, retry=retry, 
                                 request_disk=request_disk, 
                                 request_memory=request_memory, 
                                 **kwargs)

    def check_required(self):
        if not self.output_exits:
            self.required_job = True
            for p in self.parents:
                if not p.required_job:
                    p.check_required()

    def replace_input(self, cache):
        args = self.args[0].arg.split(' ')
        input_base = []
        if self.input_file:
            input_base = [os.path.basename(i_file) for i_file in self.input_file]
        for file_pair in cache:
            if file_pair[0] in input_base:
                # replace this in arguments
                for i, arg in enumerate(args):
                    if file_pair[0] in arg:
                        args[i] = file_pair[1]
        self.args = [JobArg(arg=' '.join(args),
                            name=self.args[0].name,
                            retry=self.args[0].retry)]

    def replace_input_children(self, cache):
        for c in self.children:
            c.replace_input(cache)

class Dagman(pyDagman):
    """
    Wrapper class of pycondor.Dagman
    """
    def __init__(self, name, basedir=None, **kwargs):
        self.base_dir = os.path.abspath(basedir)
        self.submit_dir = os.path.join(self.base_dir, 'condor')
        self.error_dir = os.path.join(self.base_dir, 'condor')
        self.log_dir = os.path.join(self.base_dir, 'condor')
        self.output_dir = os.path.join(self.base_dir, 'condor')
        self.result_dir = os.path.join(self.base_dir, 'output')
        # used for caching
        self.input_cache_files = []
        super(Dagman,self).__init__(name, submit=self.submit_dir, **kwargs)

    def make_dir(self):
       makedir(self.base_dir)
       makedir(self.submit_dir)
       makedir(self.error_dir)
       makedir(self.log_dir)
       makedir(self.output_dir)
       makedir(self.result_dir)

    def remove_node(self, job):
        self.nodes.remove(job)

    def remove_parent(self, child_job, parent_job):
        child_job.parents.remove(parent_job)
    
    def in_cache(self, output_files, cache_list):
        if output_files:
            return all(os.path.basename(out) in cache_list.T[0]
                       for out in output_files)
        else:
            return False
    
    def add_to_dagman_inputs(self, file_names, cache):
        for file_name in file_names:
            f_name = os.path.basename(file_name)
            f_num = np.argwhere(cache.T[0] == f_name)
            self.input_cache_files.append(cache[f_num].flatten())
    
    def cache_filter(self, cache):
        # first figure out what jobs already have their outputs
        premade_jobs = []
        # try starting with "final result" jobs
        for job in self.nodes:
            #if all(out in cache for out in job.output_file):
            if self.in_cache(job.output_file, cache):
                #premade_jobs.append(job)
                job.output_exits = True
                job.replace_input_children(cache)
                self.add_to_dagman_inputs(job.output_file, cache)
        # then sort through nodes
        for job in self.nodes:
            if job.final_result:
                job.check_required()
    
    def remove_completed_jobs(self):
        node_list = self.nodes.copy()
        for j in node_list:
            if not j.required_job:
                # first remove children
                for c in j.children:
                    self.remove_parent(c, j)
                # then remove the node
                self.remove_node(j)
    
    def filter_and_remove_from_cache(self, cache):
        self.cache_filter(cache)
        self.remove_completed_jobs()
    
    def load_cache(self, cache_file):
        cache = np.loadtxt(cache_file, dtype=str)
        return cache[list(map(os.path.isfile,cache.T[1]))]
    
    def save_output_loc(self, output_loc='pygwb_cache.txt'):
        file_names = []
        file_paths = []
        for job in self.nodes:
            if job.output_file:
                file_names += [os.path.basename(out) for out in job.output_file]
                file_paths += job.output_file
        cache_output = np.array([file_names, file_paths]).T
        if len(self.input_cache_files):
            cache_output = np.concatenate((cache_output, self.input_cache_files))
        np.savetxt(output_loc,
                   cache_output,
                   delimiter=" ", fmt="%s")

def collect_job_arguments(config, job_type):
    config_sec = config[job_type]
    args_list = list()
    for key in config_sec.keys():
        args_list.append('--' + str(key))
        val = str(config_sec[key])
        if val[0] == '$':
            val_source = val.replace('${', '').replace('}', '')
            val_source = val_source.split(':')
            val = config[val_source[0]][val_source[1]]
        args_list.append(val)
    return args_list

def create_pygwb_pipe_job(dagman, config, t0=None, tf=None, parents=[], output_path=None):
#pygwb_pipe --param_file parameters.ini --output_path output/ --t0 ${START} --tf ${END}
    dur = int(tf-t0)
    name = f'pygwb_pipe_{int(t0)}_{dur}'
    args = collect_job_arguments(config, 'pygwb_pipe')
    args = args + [
        '--output_path', output_path,
        '--t0', str(t0),
        '--tf', str(tf)
        ]
    output_file = [os.path.join(output_path, 'parameters_final.ini'),
                   os.path.join(output_path, f'psds_csds_{t0}-{tf}.npz'),
                   os.path.join(output_path, f'point_estimate_sigma_{t0}-{tf}.npz')]
    job = Job(name=name,
              executable=config['executables']['pygwb_pipe'],
              accounting_group = config['general']['accounting_group'],
              submit=dagman.submit_dir,
              error=dagman.error_dir,
              output=dagman.output_dir,
              log=dagman.log_dir,
              arguments=' '.join(args),
              request_disk='128MB',
              request_memory='2048MB',
              output_file=output_file,
              dag=dagman)
    for parent in parents:
        job.add_parent(parent)
    return job, output_file

# FIX THE JOB LIST
def create_pygwb_combine_job(dagman, config, t0=None, tf=None, parents=[], input_file=None, output_path=None,
    data_path=None, coherence_path=None, param_file=None,
    alpha=0., fref=30, h0=0.7):
#pygwb_combine --data_path output/ --param_file output/parameters_final.ini --alpha 0 --fref 30 --h0 0.7 --out_path ./output/
    dur = int(tf-t0)
    name = f'pygwb_combine_{int(t0)}_{dur}'
    args = collect_job_arguments(config, 'pygwb_combine')
    if isinstance(data_path, str):
        data_path = [data_path]
    if isinstance(coherence_path, str):
        coherence_path = [coherence_path]

    input_file = data_path + coherence_path

    data_path = ' '.join(data_path)
    coherence_path = ' '.join(coherence_path)

    args = args + [
        '--param_file', param_file,
        '--out_path', output_path,
        '--data_path', data_path,
        '--coherence_path', coherence_path,
        ]
    output_file = [os.path.join(output_path, f'point_estimate_sigma_spectra_alpha_{alpha:.1f}_fref_{int(fref)}_{t0}-{t0}.npz'),
                   os.path.join(output_path, f'coherence_spectrum_{t0}-{t0}.npz'),
                   os.path.join(output_path, f'delta_sigma_cut_{t0}-{t0}.npz')]
    job = Job(name=name,
              executable=config['executables']['pygwb_combine'],
              accounting_group = config['general']['accounting_group'],
              submit=dagman.submit_dir,
              error=dagman.error_dir,
              output=dagman.output_dir,
              log=dagman.log_dir,
              arguments=' '.join(args),
              request_disk='128MB',
              request_memory='1024MB',
              final_result=True,
              input_file=input_file,
              output_file=output_file,
              dag=dagman)
    for parent in parents:
        job.add_parent(parent)
    return job, output_file

def create_pygwb_stats_job(dagman, config, t0=None, tf=None, parents=[], output_path=None,
    csd_file=None, dsc_file=None, coh_file=None, param_file=None):
    dur = int(tf-t0)
    name = f'pygwb_stats_{int(t0)}_{dur}'
    args = collect_job_arguments(config, 'pygwb_stats')
    args = args + [
        '-pf', param_file,
        '-pd', output_path,
        '-c', csd_file,
        '-dsc', dsc_file,
        '-fcoh', coh_file
        ]
    input_file = [param_file, csd_file, dsc_file, coh_file]
    # FIXME - figure out all the plots this job makes
    output_file = None
    job = Job(name=name,
              executable=config['executables']['pygwb_stats'],
              accounting_group = config['general']['accounting_group'],
              submit=dagman.submit_dir,
              error=dagman.error_dir,
              output=dagman.output_dir,
              log=dagman.log_dir,
              arguments=' '.join(args),
              request_disk='128MB',
              request_memory='512MB',
              final_result=True,
              input_file=input_file,
              output_file=output_file,
              required_job=True,
              dag=dagman)
    for parent in parents:
        job.add_parent(parent)
    return job

def create_pygwb_html_job(dagman, config, t0=None, tf=None, parents=[], output_path=None, plot_path=None):
    name = f'pygwb_html'
    args = collect_job_arguments(config, 'pygwb_html')
    args = args + [
        '-p', plot_path,
        '-o', output_path,
        ]
    job = Job(name=name,
              executable=config['executables']['pygwb_html'],
              accounting_group = config['general']['accounting_group'],
              submit=dagman.submit_dir,
              error=dagman.error_dir,
              output=dagman.output_dir,
              log=dagman.log_dir,
              arguments=' '.join(args),
              request_disk='128MB',
              request_memory='1024MB',
              final_result=True,
              required_job=True,
              dag=dagman)
    for parent in parents:
        job.add_parent(parent)
    return job
