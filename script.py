# py_script --ini_file param.ini --t0 0 --tf 200

# log: the explicit args used
# add a logger!
from pygwb.argument_parser import parser
from pygwb.parameters import Parameters
from pathlib import Path
from pygwb.detector import Interferometer
from pygwb.baseline import Baseline

if __name__ == "__main__":
    args = parser.parse_args()
    params = Parameters.from_file(args.param_file)
    params.t0 = args.t0
    params.tf = args.tf
    outfile_path = Path(args.param_file)
    outfile_path = outfile_path.with_name(
        f"{outfile_path.stem}_final{outfile_path.suffix}"
    )
    params.save_paramfile(str(outfile_path))
    print(f"saved final param file at {outfile_path}")
    params = Parameters.from_file(outfile_path)

    ifo_H = Interferometer.from_parameters("H1", params)
    ifo_L = Interferometer.from_parameters("L1", params)

    base_HL = Baseline.from_parameters(ifo_H, ifo_L, params)
