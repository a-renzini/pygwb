#py_script --ini_file param.ini --t0 0 --tf 200

# log: the explicit args used
# add a logger!
from argument_parser import parser
from parameters import Parameters

if __name__ == "__main__":
    args = parser.parse_args()
    params=Parameters.from_file(args.param_file)
    params.t0=args.t0
    params.tf=args.tf
    params.save_new_paramfile()
    Parameters.from_file("param.ini")

    ifo_H=Interferometer.from_parameters("H1", params)
    ifo_L=Interferometer.from_parameters("L1", params)

    base_HL = Baseline.from_parameters(ifo_H, ifo_L, params)
