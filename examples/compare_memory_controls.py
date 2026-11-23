"""Compare all six conditions on one selected episode (not aggregate study)."""
import argparse
import json
from spatial_memory_kit.replay import replay
from spatial_memory_kit.reproducibility import seed_everything

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default=".")
parser.add_argument("--model", default="coupled_b7_s33101")
parser.add_argument("--route", choices=("a_one", "a_two"), default="a_two")
args = parser.parse_args()
seed_everything(0)
rows = [replay(args.root, args.model, args.route, state, history) for state in ("initial", "trained") for history in ("full", "reset-now", "reset8")]
print(json.dumps(rows, indent=2, allow_nan=False))

