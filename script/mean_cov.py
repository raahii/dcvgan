import sys

import numpy as np


def main():
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    data = np.asarray([float(l) for l in lines])
    print(data.mean(), data.std())


if __name__ == "__main__":
    main()
