import sys
import argparse
# print(sys.argv[:])

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=0)
args = parser.parse_args()
print(args.n)