import argparse

from pygwb.parameters import Parameters

parser = argparse.ArgumentParser()
parser.add_argument(
    "-param_file", help="Parameter file", action="store", type=str, required=True
)
parser.add_argument("-t0", help="Start time", action="store", type=float, default=None)
parser.add_argument("-tf", help="End time", action="store", type=float, default=None)
parser.add_argument(
    "-H1", help="Mock data for the first IFO", action="store", type=str, default=None
)
parser.add_argument(
    "-L1", help="Mock data for the second IFO", action="store", type=str, default=None
)
args = parser.parse_args()
params = Parameters.from_file(args.param_file)
if args.t0 is not None:
    params.t0 = args.t0
if args.tf is not None:
    params.tf = args.tf
if args.H1 and args.L1:
    local_data_path_dict = {"H1": args.H1, "L1": args.L1}
    params.local_data_path_dict = local_data_path_dict
