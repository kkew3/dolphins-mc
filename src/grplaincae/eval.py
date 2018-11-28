#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('filelock').setLevel(logging.WARNING)

from configparser import ConfigParser
import argparse
import re
import os
from types import SimpleNamespace

import torchvision.transforms as trans
import vmdata
from more_trans import BWCAEPreprocess
import grplaincae.viz as viz


def make_parser():
    parser = argparse.ArgumentParser(description='Run visualization task')
    parser.add_argument('cfg', help='the config filename')
    parser.add_argument('--basedir')
    return parser


def parse_cfg(basedir: str, filename: str) -> SimpleNamespace:
    runid = int(re.findall(r'\b(\d+)\b', os.path.splitext(
        os.path.basename(filename))[0])[0])
    ini = ConfigParser()
    ini.read(filename)
    ns = SimpleNamespace()

    camera_id = list(map(int, map(str.strip, ini['data']['camera'].split(','))))
    ns.root = vmdata.prepare_dataset_root(camera_id[0], tuple(camera_id[1:4]))
    ns.downsample_scale = int(ini['data']['downsample_scale'].strip())
    ns.model_name = ini['model']['model_name'].strip()
    run_name = ini['model']['run_name'].strip()
    ns.savedir = os.path.join(basedir, '{}.{}'.format(run_name, runid), 'save')
    ns.progress = tuple(map(int, map(str.strip, ini['model']['progress'].split(','))))
    ns.visualize_indices = range(*tuple(map(int, map(
        str.strip, ini['eval']['indices'].split(',')))))
    try:
        temperature = float(ini['eval']['temperature'].strip())
    except (KeyError, ValueError):
        temperature = 1.0
    ns.temperature = temperature
    try:
        bwth = float(ini['eval']['bwth'].strip())
    except (KeyError, ValueError):
        bwth = None
    ns.bwth = bwth
    ns.todir = os.path.join(basedir,
                            '{}.{}.viz'.format(run_name, runid),
                            '{}_{}'.format(*ns.progress),
                            't{:.2f}'.format(ns.temperature))
    try:
        device = ini['eval']['device'].strip()
    except KeyError:
        device = 'cpu'
    ns.device = device
    try:
        batch_size = int(ini['eval']['batch_size'].strip())
    except KeyError:
        batch_size = 8
    ns.batch_size = batch_size
    return ns


def main():
    args = make_parser().parse_args()
    ns = parse_cfg(os.getcwd() if not args.basedir else args.basedir, args.cfg)
    net, m = viz.load_trained_net(ns.model_name, ns.savedir, ns.progress)
    bw = (m.input_color == 'gray')
    normalize_stats = vmdata.get_normalization_stats(ns.root, bw=bw)
    assert bw, 'Illegal bw ({})'.format(bw)
    transform = BWCAEPreprocess(trans.Normalize(*normalize_stats),
                                pool_scale=m.pool_scale,
                                downsample_scale=ns.downsample_scale)
    logging.info('Complete initialization: config={}'.format(ns))
    viz.visualize(ns.todir, ns.root, transform, normalize_stats,
                  ns.visualize_indices, net, m, temperature=ns.temperature,
                  bwth=ns.bwth, device=ns.device, batch_size=ns.batch_size)


if __name__ == '__main__':
    main()
