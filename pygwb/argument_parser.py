import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-param_file", help="Parameter file", action="store", type=str, required=True
)
parser.add_argument("-t0", help="Start time", action="store", type=float, default=None)
parser.add_argument("-tf", help="End time", action="store", type=float, default=None)
