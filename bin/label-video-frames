#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt

import vmdata


def make_parser():
    parser = argparse.ArgumentParser(description='Label video frames. The '
                                                 'labels are exported to a '
                                                 'text file, one label per '
                                                 'line.')
    parser.add_argument('root', help='the video dataset root')
    parser.add_argument('outfile', help='file to which the labels are exported')
    return parser

def cmdloop(vdset, filename):
    plt.close('all')
    plt.figure()
    plt.ion()
    plt.show()
    labels = []

    for i in range(len(vdset)):
        plt.imshow(vdset[i])
        plt.draw()
        plt.pause(1e-3)

        while True:
            sbuf = input('{}/{} label> '.format(i, len(vdset))).strip()
            if sbuf:
                labels.append(sbuf)
                break

    with open(filename, 'w') as outfile:
        outfile.write('\n'.join(labels) + '\n')


if __name__ == '__main__':
    args = make_parser().parse_args()
    with vmdata.VideoDataset(args.root) as dset:
        cmdloop(dset, args.outfile)
