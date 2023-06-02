# -*- coding: utf-8 -*-
# Copyright (C) California Institute of Technology 2023
#
# This file is part of pygwb.

import argparse
import glob
import logging
import os

import gwsumm
import gwsumm.plot
import gwsumm.tabs
from gwpy.time import from_gps

"""Create pygwb html
"""

def pygwb_html(outdir='./', config=None):

    # =====================
    # get data
    
    ifo = 'Network'
    # =====================
    # now make the html page

    os.chdir('/'.join(outdir.split('/')[:-1]))
    output_dir = outdir.split('/')[-1]
    
    tabs = []
    base_tabs = []
    all_plots = []
    
    # now make main tab
    logging.info('Processing main results page...')
    home_index = os.path.join(output_dir,'index.html')

    plot_sub_dir = os.path.join(output_dir, 'output/combined_results')
    home_plots = glob.glob(plot_sub_dir+'/*.png')
    home_plots = sorted(home_plots)
    home_plots = ['/'.join(plot_loc.split('/')[-3:]) for plot_loc in home_plots]
    
    plot_tab = gwsumm.tabs.PlotTab('Stochmon combined results',path=output_dir,index=home_index)
    plot_tab.set_layout((2,2))
    for plot_name in home_plots:
        plot = gwsumm.plot.core.SummaryPlot(href='./'+os.path.join(output_dir,plot_name))

        plot_tab.add_plot(plot)
    base_tabs.append(plot_tab)
    tabs.append(plot_tab)

    seg_tabs = []
    segment_folders = glob.glob(output_dir+'/output/*-*')
    for i, seg_folder in enumerate(segment_folders):
        page_name = seg_folder.split('/')[-1].split('-')[0]
        page_name = from_gps(int(page_name)).strftime('%Y-%m-%d %H:%M:%S')
        if i==0:
            tab0 = gwsumm.tabs.PlotTab(page_name,
                                       path=output_dir,parent='Segment results')
            tab = tab0
        else:
            tab = gwsumm.tabs.PlotTab(page_name,
                                      path=output_dir,parent=tab0.parent)

        seg_plots = glob.glob(seg_folder+'/*.png')
        seg_plots = sorted(seg_plots)
        seg_plots = ['/'.join(plot_loc.split('/')[-3:]) for plot_loc in seg_plots]
        for plot_name in seg_plots:
            plot = gwsumm.plot.core.SummaryPlot(href='./'+os.path.join(output_dir,plot_name))
            tab.add_plot(plot)
        tab.set_layout((2))
        seg_tabs.append(tab)
        tabs.append(tab)

    
    about_tab = gwsumm.tabs.AboutTab('About',path=output_dir)
    about_href = about_tab.href
    
    # Write out tabs
    logging.info('Writing html of tabs...')
    for tab in base_tabs:
        tab.write_html('',tabs=tabs,about=about_href)

    for tab in seg_tabs:
        tab.write_html('',tabs=tabs,about=about_href)
    
    if config:
        about_tab.write_html(tabs=tabs, config=config) # fix me

def main(args=None):
    logging.basicConfig()
    logger = logging.getLogger(__process_name__)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__,
                                     prog=__process_name__)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase verbose output')
    parser.add_argument('-V', '--version', action='version',
                        version=__version__)
    parser.add_argument('-o', '--output-dir', type=os.path.abspath,
                        help='Directory for all output')
    parser.add_argument('-p', '--plot-dir', type=os.path.abspath,
                        help='Directory of plots to show')
    args = parser.parse_args(args=args)

    # call the above function
    pygwb_html(outdir=args.output_dir)

# allow be be run on the command line
if __name__ == "__main__":
    main(args=None)

