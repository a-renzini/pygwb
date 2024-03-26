# -*- coding: utf-8 -*-
# Copyright (C) California Institute of Technology 2023
#
# This file is part of pygwb.

"""Core workflow functions"""

import configparser
import logging
import os
import re
import shutil
from collections.abc import Iterable
from getpass import getuser

import numpy as np
from gwpy.io.cache import cache_segments
from gwpy.segments import DataQualityDict, DataQualityFlag, Segment, SegmentList
from gwsumm.data.timeseries import find_best_frames
from pycondor import Dagman as pyDagman
from pycondor import Job as pyJob
from pycondor.job import JobArg

ACCOUNTING_GROUP_USER = os.getenv(
    '_CONDOR_ACCOUNTING_USER',
    getuser(),
)

def _split(orig_list, N):
    k, m = divmod(len(orig_list), N)
    return list(orig_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(N))

def makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
                f"getenv = True",
                ]
            condorcmds.extend(extra_lines)
        else:
            raise TypeError('accounting group must be supplied')

        output_args = []
        self.output_file = None
        # attrs for cache system
        self.output_exits = False
        self.required_job = False
        self.input_replaced = False
            
        # output args
        if output_file is not None:
            if output_arg:
                if isinstance(output_arg, str):
                    output_args.append(output_arg)
                else:
                    raise TypeError('output_arg must be a string')
            if isinstance(output_file, str):
                self.output_file = [output_file]
                if output_arg:
                    output_args.append(output_file)
            elif isinstance(output_file, Iterable):
                self.output_file = output_file
                if output_arg:
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

        # surround dicts with ""
        # THIS HAS BEEN SOMEHOW DEPRECATED: CONDOR ADDS THEM AUTOMATICALLY
        #arguments = arguments.replace("{", "'{").replace("}", "}'")

        # convert memory into integer
        try:
            request_memory = int(request_memory)
        except:
            # check if it is MB, GB, or KB
            if 'M' in request_memory:
                request_memory = int(re.findall('\d+', request_memory)[0])
            elif 'G' in request_memory:
                request_memory = int(re.findall('\d+', request_memory)[0]) * 1024
            else:
                raise TypeError('request_memory must be an integer or in format of e.g. 1024MB or 1GB')

        # release if requested memory is too small
        request_memory = f"ifthenelse(isUndefined(MemoryUsage),{request_memory},3*MemoryUsage)"
        condorcmds.extend(["periodic_release = (HoldReasonCode == 26) && (JobStatus == 5)"])

        if retry == None:
            retry = 3
            
        super(Job,self).__init__(name, exec_path, arguments=arguments,
                                 extra_lines=condorcmds, retry=retry, 
                                 request_disk=request_disk, 
                                 request_memory=request_memory, 
                                 **kwargs)

    def check_required(self):
        if not self.output_exits or self.final_result:
            self.required_job = True
            for p in self.parents:
                if not p.required_job:
                    p.check_required()

    def replace_input(self, cache):
        if not self.input_replaced:
            args = self.args[0].arg.split(' ')
            input_base = []
            if self.input_file:
                input_base = [os.path.basename(i_file) for i_file in self.input_file]
            for file_base in input_base:
                file_index = np.where(cache.T[0] == file_base)
                if len(file_index[0]):
                    # replace this in arguments
                    for i, arg in enumerate(args):
                        if file_base in arg:
                            args[i] = cache.T[1][file_index[0]][0]
            self.args = [JobArg(arg=' '.join(args),
                                name=self.args[0].name,
                                retry=self.args[0].retry)]
            self.input_replaced = True

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

    def run_jobs_serially(self):
        import subprocess

        # run everything locally and serially
        remaining_nodes = self.nodes.copy()
        node_order = []
        node_step = 0

        # first add jobs with no parents
        for node in remaining_nodes:
            if not len(node.parents):
                node_order.append([node.name, node.executable, node.args[0].arg])
                remaining_nodes.remove(node)

        # now check the other nodes
        while len(remaining_nodes):
            for node in remaining_nodes:
                if all([n not in remaining_nodes for n in node.parents]):
                    node_order.append([node.name, node.executable, node.args[0].arg])
                    remaining_nodes.remove(node)
            node_step += 1
            #double check that we aren't in an inf loop
            if node_step > 100:
                raise RuntimeError('Too many steps when trying '
                                   'to run serially!')

        logging.info('Beginning to run all jobs serially...')
        node_num = 1
        node_len = len(node_order)
        for node in node_order:
            path_and_args = node[1] + ' ' + node[2]
            logging.info(f'Starting node #{node_num}/{node_len}: {node[0]}')
            logging.info(f'The full path and arguments are: {path_and_args}')
            subprocess.run(path_and_args, check=True, shell=True)
            node_num += 1
        logging.info('Workflow completed!')

def _collect_job_arguments(config, job_type):
    config_sec = config[job_type]
    args_list = list()
    for key in config_sec:
        args_list.append('--' + str(key))
        val = str(config_sec[key])
        if val[0] == '$':
            val_source = val.replace('${', '').replace('}', '')
            val_source = val_source.split(':')
            val = config[val_source[0]][val_source[1]]
        args_list.append(val)
    return args_list

class Workflow():
    def __init__(self, name, config_file, basedir='./workflow', 
                 config_overrides=[]):
        self.t0 = None
        self.tf = None

        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.config_override(config_overrides)

        self.cache_file_path = None
        if self.config.has_option('general', 'cache_file'):
            self.cache_file_path = os.path.abspath(self.config['general']['cache_file'])

        self.setup_dagman(name, basedir=basedir) 

    def config_override(self, overrides):
        for ov in overrides:
            split_ov = ov.split(':')
            try:
                ov_sec, ov_key, ov_val = \
                    split_ov[0], split_ov[1], ':'.join(split_ov[2:])
            except:
                raise ValueError(f"Overrides must be in the form 'section:key:value', you used '{ov}'")
            if ov_sec not in self.config.sections():
                self.config[ov_sec] = {}
            self.config[ov_sec][ov_key] = ov_val

    def config_write(self, output_loc):
        with open(output_loc, 'w') as configfile:
            self.config.write(configfile)

    def setup_segments(self):
        self.t0 = int(self.config['general']['t0'])
        self.tf = int(self.config['general']['tf'])
        dur = int(self.tf-self.t0)

        if self.config.has_option('general', 'min_job_dur'):
            self.min_dur = int(self.config['general']['min_job_dur'])
        else:
            self.min_dur = 0
        if self.config.has_option('general', 'max_job_dur'):
            self.max_dur = int(self.config['general']['max_job_dur'])
        else:
            self.max_dur = 1000000

        ifos = self.config['general']['ifos'].split(' ')

        logging.info('Downloading science flags...')
        if self.config.has_option('data_quality', 'science_segment'):
            sci_flag = DataQualityDict.query_dqsegdb(
                [i+':'+self.config['data_quality']['science_segment'] for i in ifos],
                self.t0, self.tf
                ).intersection().active
        else:
            # just use provided start and end time
            sci_flag = SegmentList([[self.t0, self.tf]])

        if self.config.has_option('data_quality', 'veto_definer'):
            veto_definer_path = self.config['data_quality']['veto_definer']
            local_veto_definer_path = os.path.join(self.dagman.base_dir, 'vetoes.xml')
            if 'http' in veto_definer_path: # download vdf first
                import ciecplib
                with ciecplib.Session() as s:
                    s.get("https://git.ligo.org/users/auth/shibboleth/callback")
                    r = s.get(veto_definer_path, allow_redirects=True)
                    r.raise_for_status()
                output_fp = open(local_veto_definer_path, "wb")
                output_fp.write(r.content)
                output_fp.close()
                veto_definer_path = local_veto_definer_path

            logging.info('Downloading vetoes...')
            try:
                vetoes = DataQualityDict.from_veto_definer_file(
                    veto_definer_path
                    )
            except:
                raise TypeError('Unable to read veto definer! '
                                'This may be to an improperly formatted file '
                                'or not correctly authenticating when the veto definer '
                                'is provided as a url.'
                               )
            cat1_vetoes = DataQualityDict({v: vetoes[v] for v in vetoes if (vetoes[v].category ==1) and (v[:2] in ifos)})
            cat1_vetoes.populate()

            cat1_vetoes = cat1_vetoes.union().active
            seglist = sci_flag - cat1_vetoes

        else:
            seglist = sci_flag

        seglist_pruned = []
        for seg in seglist:
            start = int(seg[0])
            end = int(seg[1])
            dur = int(end-start)
            if dur < self.min_dur:
                continue
            elif dur > self.max_dur:
                edges = np.arange(start, end, self.max_dur)
                if edges[-1]+self.min_dur<end:
                    edges = np.append(edges, end)
                for i in range(len(edges)-1):
                    seglist_pruned.append([int(edges[i]),int(edges[i+1])])
            else:
                seglist_pruned.append([start, end])
        return seglist_pruned

    def setup_dagman(self, name, basedir=None):
        logging.info('Writing DAG...')
        self.dagman = Dagman(name=name,
                        basedir=basedir)
        self.dagman.make_dir()

        self.config_path = os.path.join(self.dagman.base_dir, 'config.ini')
        self.config_write(self.config_path)

class IsotropicWorkflow(Workflow):
    def __init__(self, name, config_file, basedir='./workflow',
                 config_overrides=[]):
        super(IsotropicWorkflow, self).__init__(name, 
                                                config_file, basedir=basedir,
                                                config_overrides=config_overrides)
        self.fref = int(self.config['pygwb_pipe']['fref'])
        self.alpha =  float(self.config['pygwb_pipe']['alpha'])

    def create_pygwb_pipe_job(self, t0=None, tf=None, parents=[], output_path=None):
    #pygwb_pipe --param_file parameters.ini --output_path output/ --t0 ${START} --tf ${END}
        dur = int(tf-t0)
        file_tag = f'{int(t0)}-{dur}'
        name = f'pygwb_pipe_{int(t0)}_{dur}'
        args = _collect_job_arguments(self.config, 'pygwb_pipe')
        args = args + [
            '--output_path', output_path,
            '--t0', str(t0),
            '--tf', str(tf),
            '--file_tag', file_tag
            ]
        output_file = [os.path.join(output_path, f'parameters_{file_tag}_final.ini'),
                       os.path.join(output_path, f'psds_csds_{file_tag}.npz'),
                       os.path.join(output_path, f'point_estimate_sigma_{file_tag}.npz')]
        job = Job(name=name,
                  executable=self.config['executables']['pygwb_pipe'],
                  accounting_group = self.config['general']['accounting_group'],
                  submit=self.dagman.submit_dir,
                  error=self.dagman.error_dir,
                  output=self.dagman.output_dir,
                  log=self.dagman.log_dir,
                  arguments=' '.join(args),
                  request_disk='128MB',
                  request_memory='2048MB',
                  output_file=output_file,
                  dag=self.dagman)
        for parent in parents:
            job.add_parent(parent)
        return job, output_file
    
    # FIX THE JOB LIST
    def create_pygwb_combine_job(self, t0=None, tf=None, parents=[], input_file=None, output_path=None,
        data_path=None, coherence_path=None, dsc_path=None, param_file=None,
        alpha=0., fref=30, h0=0.7, file_tag_extra=None):
    #pygwb_combine --data_path output/ --param_file output/parameters_final.ini --alpha 0 --fref 30 --h0 0.7 --out_path ./output/
        dur = int(tf-t0)
        file_tag = f'{int(t0)}-{dur}'
        name = f'pygwb_combine_{int(t0)}_{dur}'
        if file_tag_extra:
            file_tag = f'{file_tag}-{file_tag_extra}'
            name = f'{name}_{file_tag_extra}'
        args = _collect_job_arguments(self.config, 'pygwb_combine')
        if isinstance(data_path, str):
            data_path = [data_path]
        if isinstance(coherence_path, str):
            coherence_path = [coherence_path]
    
        input_file = data_path + coherence_path + [param_file]

        data_path = ' '.join(data_path)
        coherence_path = ' '.join(coherence_path)
    
        args = args + [
            '--param_file', param_file,
            '--out_path', output_path,
            '--data_path', data_path,
            '--coherence_path', coherence_path,
            '--file_tag', file_tag
            ]

        if dsc_path:
            # only provided for the "combine combine" jobs
            if isinstance(dsc_path, str):
                dsc_path = [dsc_path]
            input_file = input_file + dsc_path
            dsc_path = ' '.join(dsc_path)
            args = args + [
                '--delta_sigma_path', dsc_path,
                ]

        output_file = [os.path.join(output_path, f'point_estimate_sigma_spectra_alpha_{alpha:.1f}_fref_{int(fref)}_{file_tag}.npz'),
                       os.path.join(output_path, f'coherence_spectrum_{file_tag}.npz'),
                       os.path.join(output_path, f'delta_sigma_cut_{file_tag}.npz')]
        job = Job(name=name,
                  executable=self.config['executables']['pygwb_combine'],
                  accounting_group = self.config['general']['accounting_group'],
                  submit=self.dagman.submit_dir,
                  error=self.dagman.error_dir,
                  output=self.dagman.output_dir,
                  log=self.dagman.log_dir,
                  arguments=' '.join(args),
                  request_disk='128MB',
                  request_memory='1024MB',
                  final_result=True,
                  input_file=input_file,
                  output_file=output_file,
                  dag=self.dagman)
        for parent in parents:
            job.add_parent(parent)
        return job, output_file
    
    def create_pygwb_stats_job(self, t0=None, tf=None, parents=[], output_path=None,
        csd_file=None, dsc_file=None, coh_file=None, param_file=None):
        dur = int(tf-t0)
        name = f'pygwb_stats_{int(t0)}_{dur}'
        args = _collect_job_arguments(self.config, 'pygwb_stats')
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
                  executable=self.config['executables']['pygwb_stats'],
                  accounting_group = self.config['general']['accounting_group'],
                  submit=self.dagman.submit_dir,
                  error=self.dagman.error_dir,
                  output=self.dagman.output_dir,
                  log=self.dagman.log_dir,
                  arguments=' '.join(args),
                  request_disk='128MB',
                  request_memory='512MB',
                  final_result=True,
                  input_file=input_file,
                  output_file=output_file,
                  required_job=True,
                  dag=self.dagman)
        for parent in parents:
            job.add_parent(parent)
        return job

    def create_pygwb_multi_stage_combine(self, t0=None, tf=None, parents=[], input_file=None, output_path=None,
        data_path=None, coherence_path=None, dsc_path=None, param_file=None,
        alpha=0., fref=30, h0=0.7, combine_factor=1):

        # set up individual combines
        assert len(data_path) == len(coherence_path)
        data_path_lists = _split(data_path, int(combine_factor))
        coherence_path_lists = _split(coherence_path, int(combine_factor))

        combined_data_path = []
        combined_coherence_path = []
        combined_dsc_path = []
        combined_jobs = []

        for i in range(int(combine_factor)):
            extra_tag = f'PART{int(i)}'
            combine_job, combine_output = self.create_pygwb_combine_job(
                t0=self.t0, tf=self.tf,
                parents=parents, output_path=output_path,
                data_path=data_path_lists[i], coherence_path=coherence_path_lists[i], param_file=param_file,
                alpha=self.alpha, fref=self.fref, file_tag_extra=extra_tag)
            combined_data_path.append(combine_output[0])
            combined_coherence_path.append(combine_output[1])
            combined_dsc_path.append(combine_output[2])
            combined_jobs.append(combine_job)

        #now the big combine
        final_job, output_file = self.create_pygwb_combine_job(
            t0=self.t0, tf=self.tf,
            parents=combined_jobs+[parents[-1]], output_path=output_path,
            data_path=combined_data_path, coherence_path=combined_coherence_path, param_file=param_file,
            dsc_path=combined_dsc_path,
            alpha=self.alpha, fref=self.fref)
        combined_jobs.append(final_job)

        return combined_jobs, final_job, output_file
    
    def create_pygwb_html_job(self, t0=None, tf=None, segment_results=False, parents=[], output_path=None, plot_path=None):
        name = f'pygwb_html'
        args = _collect_job_arguments(self.config, 'pygwb_html')
        args = args + [
            '-p', plot_path,
            '-o', output_path,
            ]
        if segment_results:
            args = args + ['--plot-segment-results']
        job = Job(name=name,
                  executable=self.config['executables']['pygwb_html'],
                  accounting_group = self.config['general']['accounting_group'],
                  submit=self.dagman.submit_dir,
                  error=self.dagman.error_dir,
                  output=self.dagman.output_dir,
                  log=self.dagman.log_dir,
                  arguments=' '.join(args),
                  request_disk='128MB',
                  request_memory='1024MB',
                  final_result=True,
                  required_job=True,
                  dag=self.dagman)
        for parent in parents:
            job.add_parent(parent)
        return job
