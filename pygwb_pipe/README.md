# Running the pygwb_pipe pipeline

make sure you have installed the package (pip install .) and all requirements are met.

run

```
./pygwb_pipe/pygwb_pipe --param_file ./pygwb_pipe/parameters.ini --apply_dsc False
```
if you would like to specify a start and end time, `--t0` and `--tf` are supported arguments.

You should be getting (for the default value of h0 = 0.6932),

```
POINT ESTIMATE: -6.189551e-06
SIGMA: 2.561543e-06
```

If you get a different result *with the same ini file*, something meaningful has been changed in the pipe! 
