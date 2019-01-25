#!/usr/bin/env python3
import argparse
import sys


def make_parser():
    parser = argparse.ArgumentParser(
        description='Converts a column of numbers into ranges separated by '
                    'dashes and commas. The numbers will be read from stdin. '
                    'Each number in the input should be unique and the '
                    'numbers should have been sorted in ascending order. '
                    'Example: given input \'1\\n2\\n3\\n5\\n\', the output '
                    'will be \'1-3\\n5\\n\'')
    parser.add_argument('-T', '--tol', type=int, default=0,
                        help='tolerance of gap between successive numbers; '
                             'default to %(default)s')
    parser.add_argument('-M', '--min-len', type=int, default=1, dest='mlen',
                        help='minimum length of successive numbers; default '
                             'to %(default)s')
    parser.add_argument('-j', '--join2', action='store_true',
                        help='join ranges of length 2 by dash rather than '
                             'write the two numbers in two lines; default to '
                             '%(default)s')
    parser.add_argument('-s', '--sort', action='store_true',
                        help='sort the input before conversion; note that '
                             'this option prevents the script from running '
                             'in a streaming manner; default to %(default)s')
    return parser


def nums2ranges(sorted_uniq_nums, tol=0, mlen=1):
    cur_rng = None
    for x in sorted_uniq_nums:
        if cur_rng is None:
            cur_rng = [x, 1]
        elif x <= cur_rng[0] + cur_rng[1] + tol:
            cur_rng[1] += 1
        else:
            if cur_rng[1] >= mlen:
                yield (cur_rng[0], cur_rng[0] + cur_rng[1] - 1)
            cur_rng = [x, 1]
    if cur_rng:
        if cur_rng[1] >= mlen:
            yield (cur_rng[0], cur_rng[0] + cur_rng[1] - 1)


def format_rng(rng, j2=False):
    if rng[0] == rng[1]:
        sbuf = [str(rng[0])]
    elif rng[0] + 1 == rng[1] and not j2:
        sbuf = map(str, rng)
    elif rng[0] + 1 == rng[1] and j2:
        sbuf = ['{}-{}'.format(*map(str, rng))]
    else:
        sbuf = ['{}-{}'.format(*map(str, rng))]
    return '\n'.join(sbuf)


if __name__ == '__main__':
    args = make_parser().parse_args()
    if sys.stdin.isatty():
        print('Expecting numbers from stdin', file=sys.stderr)
        sys.exit(1)
    nums = map(int, filter(None, map(str.strip, sys.stdin)))
    if args.sort:
        nums = sorted(set(nums))
    for r in nums2ranges(nums, tol=args.tol, mlen=args.mlen):
        print(format_rng(r, j2=args.join2))
