#!/usr/bin/env python3
import argparse
import os


def make_parser():
    parser = argparse.ArgumentParser(
        description='Generate YAMLs for Scalabel tasks. The gneerated YAML '
                    'will be named "$(dirname "$IMGDIR")/_'
                    '$(basename "$IMGDIR").Scalabel-${TASK}[_${batch}].yml".')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true')
    subparsers = parser.add_subparsers(dest='task')
    parser_i = subparsers.add_parser('items', help='generate item list')

    parser_i.add_argument('basedir', metavar='BASEDIR', type=os.path.normpath,
                          help='the directory serving as the root of the '
                               'image server')
    parser_i.add_argument('imgdir', metavar='IMGDIR', type=os.path.normpath,
                          help='the directory holding the frames; must be a '
                               'child directory of BASEDIR')
    parser_i.add_argument('-u', '--baseurl', default='http://localhost:8000/',
                          help='default to %(default)s')
    parser_i.add_argument('-n', '--batchsize', type=int,
                          help='max number of images in each batch')

    parser_c = subparsers.add_parser(
        'categories', help='generate category list')
    parser_c.add_argument('imgdir', metavar='IMGDIR', type=os.path.normpath,
                          help='the directory holding the frames')
    return parser


def get_tofile(imgdir, task, batch=None):
    tokens = [os.path.dirname(imgdir), '/_',
              os.path.basename(imgdir),
              '.Scalabel-', task, '.yml']
    if batch is not None:
        tokens.insert(-1, '_{}'.format(batch))
    return ''.join(tokens)


def read_item_list(imgdir):
    imgnames = [(os.path.splitext(x)[0], x) for x in os.listdir(imgdir)]
    imgnames.sort(key=lambda x: x[0])
    return [x[1] for x in imgnames]


def write_line_item(basedir, imgpath, baseurl):
    imgpath = os.path.relpath(imgpath, start=basedir)
    imgpath = imgpath.replace('[', '%5B').replace(']', '%5D')
    assert not imgpath.startswith('/')
    imgurl = baseurl + imgpath
    return "- {{url: '{}'}}\n".format(imgurl)


if __name__ == '__main__':
    args = make_parser().parse_args()
    linesbuf = []
    if args.task == 'items':
        imgnames = read_item_list(args.imgdir)
        for name in imgnames:
            imgpath = os.path.join(args.imgdir, name)
            line = write_line_item(args.basedir, imgpath, args.baseurl)
            linesbuf.append(line)
    elif args.task == 'categories':
        linesbuf.append('- name: dolphin\n')

    if args.dry_run:
        if args.batchsize is None:
            tofile = get_tofile(args.imgdir, args.task)
            print('tofile: "{}" =>'.format(tofile))
            for l in linesbuf:
                print(l, end='')
        else:
            for bid, si in enumerate(range(0, len(linesbuf), args.batchsize)):
                tofile = get_tofile(args.imgdir, args.task, batch=bid)
                print('tofile: "{}" =>'.format(tofile))
                for l in linesbuf[si:si + args.batchsize]:
                    print(l, end='')
    else:
        if args.batchsize is None:
            tofile = get_tofile(args.imgdir, args.task)
            with open(tofile, 'w') as outfile:
                for l in linesbuf:
                    outfile.write(l)
        else:
            for bid, si in enumerate(range(0, len(linesbuf), args.batchsize)):
                tofile = get_tofile(args.imgdir, args.task, batch=bid)
                with open(tofile, 'w') as outfile:
                    for l in linesbuf[si:si + args.batchsize]:
                        outfile.write(l)
