#!/usr/bin/env python
if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('ul', type=int, help='upper limit')
    parser.add_argument('size', nargs='+', type=int)
    args = parser.parse_args()
    a = np.random.randint(args.ul, size=tuple(args.size))
    a = str(a.tolist()).replace(*'[{').replace(*']}').replace('}, {','},\n{')
    with open(*'ow') as out:
        out.write(a)
