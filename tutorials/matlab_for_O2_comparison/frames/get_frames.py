import gwpy.timeseries

tstart = 1184098680 # start time of job, here: https://git.ligo.org/stochastic-public/stochastic/-/blob/O2StochIso/example_output_for_comparison.txt
tend = 1184101352
dur = 2672

data_H1=gwpy.timeseries.TimeSeries.fetch_open_data('H1',tstart,tend)
data_L1=gwpy.timeseries.TimeSeries.fetch_open_data('L1',tstart,tend)

data_H1.channel='H1:STRAIN' # pipeline likes it when channel starts with ifo
data_L1.channel='L1:STRAIN'

frame_H = 'H-STRAIN-%d-%d.gwf'%(tstart,dur)
frame_L = 'L-STRAIN-%d-%d.gwf'%(tstart,dur)

data_H1.write(frame_H)
data_L1.write(frame_L)
