# Running the pygwb_pipe pipeline

make sure you have installed the package (pip install .) and all requirements are met.

run

```
./pygwb_pipe -param_file parameters.ini
```
if you would like to specify a start and end time, `-t0` and `t_f` are supported arguments.

You should be getting

```
POINT ESIMATE: -2.966117e-06
SIGMA: 1.229361e-06
```

If you get a different result *with the same ini file*, something meaningful has been changed in the pipe! 

