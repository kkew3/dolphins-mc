#!/usr/bin/env python
import argparse
from typing import Tuple
import readline
import cmd
import re
import sys

from PIL import Image, ImageDraw
import cv2

description = '''
Mark pixel with a red cross on an image IMG.

Two working modes: 1) command mode, when the coordinate of the pixel is given
via option '-c'; and 2) interactive mode, when it's not given. In interactive
mode, at each prompt, the user is expected to give the coordinate in a
space-separated list of two nonnegative integers. The interactive mode won't
work unless connected to a display. Option '-o' is ignored under interactive
mode.
'''

description = ' '.join(filter(lambda x: x, map(str.strip, description.split('\n'))))


def make_parser():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', dest='outfile',
                        help='file to write; if not specified, display the '
                             'annotated image directly')
    parser.add_argument('-r', dest='radius', type=int, default=1,
                        help='the radius of the cross to annotate')
    parser.add_argument('-c', dest='coordinate', metavar=('X', 'Y'), nargs=2,
                        type=int,
                        help='the coordinate to mark the location; if not '
                             'given, the program will enter interactive mode')
    parser.add_argument('img', metavar='IMG',
                        help='the image (readonly unless -o to this file) to '
                             'annotate')
    return parser


def annotate(coordinate: Tuple[int, int], img: Image, cross_radius: int) -> Image:
    """
    Annotate ``img`` with a red cross.

    :param coordinate: the pixel coordinate to locate
    :type coordinate: Tuple[int, int]
    :param img: the PIL image to annotate
    :type img: PIL.Image
    :return: the annotated copy
    :rtype: PIL.Image
    """
    img = img.copy()
    x, y = coordinate
    w, h = img.size
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError()

    offset_range = [-cross_radius, cross_radius]
    draw = ImageDraw.Draw(img)
    for xoffset in offset_range:
        for yoffset in offset_range:
            ex = x + xoffset
            ey = y + yoffset
            if 0 <= ex < w and 0 <= ey < h:
                draw.line([(ex, ey), coordinate], fill='red')
    del draw
    return img


import matplotlib.pyplot as plt
import numpy as np

def draw_bbox(ul: np.ndarray, br: np.ndarray, img: Image) -> Image:
    """
    Draw bounding box from the upper left coordinate ``ul`` and the bottom
    left coordinate ``br`` on a copy of image ``img``.

    :param ul: the upper left coordinate
    :type ul: np.ndarray
    :param br: the bottom right coordinate
    :type br: np.ndarray
    :param img: the image to annotate
    :type img: PIL.Image
    :return: the annotated copy
    :rtype: PIL.Image
    """
    xy = [tuple(ul), tuple(br + np.ones_like(br))]
    img = np.asarray(img.copy())
    cv2.rectangle(img, xy[0], xy[1], (255, 0, 0), 2)
    img = Image.fromarray(img)
    return img

args = make_parser().parse_args()
img = Image.open(args.img)
if args.coordinate:
    args.coordinate = tuple(args.coordinate)
    try:
        aimg = annotate(args.coordinate, img, args.radius)
    except ValueError:
        print('*** Pixel {} out of image'.format(args.coordinate),
              file=sys.stderr)
        sys.exit(1)
    else:
        if args.outfile:
            img.save(args.outfile)
        else:
            plt.imshow(np.asarray(aimg))
            plt.show()
        sys.exit(0)

import matplotlib.pyplot as plt
import numpy as np

app_intro = '''
At each prompt, enter coordinate (X,Y) separated by one or more spaces. Both
X and Y should be nonnegative integers. There're commands other than typing the
coordinate directly. To list all available commands, type ?. To show help for
command COMMAND, type ?COMMAND.
'''.strip()


class CmdApp(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = 'coordinate> '
        # legal line pattern after strip
        self.cpat = re.compile(r'^\+?(\d+)[ \t]+\+?(\d+)$')
        self.intro = app_intro
        self.recorded_coordinates = []
        self.prev_coordinate = None
        self.quiet = False

    def emptyline(self):
        pass

    def parse_coordinate(self, line):
        matched = self.cpat.match(line)
        if not matched:
            raise ValueError()
        return int(matched.group(1)), int(matched.group(2))

    def updage_canvas(self, new_img: Image):
        plt.imshow(np.asarray(new_img))
        plt.draw()
        plt.pause(1e-3)

    def default(self, line):
        """
        Usage: X Y
        """
        try:
            coordinate = self.parse_coordinate(line)
        except ValueError:
            print('*** Illegal coordinate pair')
            return

        self.prev_coordinate = coordinate
        try:
            aimg = annotate(coordinate, img, args.radius)
        except ValueError:
            print('*** Pixel {} out of image'.format(coordinate))
            return

        self.updage_canvas(aimg)

    def do_s(self, line):
        """
        Same as typing the coordinate directly.

        Usage: s X Y
        """
        self.default(line)

    def print_recorded_coordinates(self):
        print('Recorded coordinate (one per row):')
        print(np.stack(self.recorded_coordinates))

    def do_record(self, line):
        """Record the previous coordinate; no argument is required.
        """
        if self.prev_coordinate is None:
            print('*** No previous coordinate')
        else:
            coordinate = np.array(self.prev_coordinate, dtype=np.int32)
            self.recorded_coordinates.append(coordinate)
            if not self.quiet:
                self.print_recorded_coordinates()

    def do_r(self, line):
        """
        Alias of `record'.
        """
        self.do_record(line)

    def do_pr(self, line):
        """
        Print recorded coordinates.

        Usage: pr
        """
        if self.recorded_coordinates:
            self.print_recorded_coordinates()
        else:
            print('*** No recorded coordinates')

    def do_delete(self, line):
        """
        Delete the Nth (default to the last) recorded coordinate.

        Usage: delete [N]
        Example 1 (delete the last record): delete
        Example 2 (delete the first record): delete 0
        Example 3 (delete the second record): delete 1
        """
        if self.recorded_coordinates:
            try:
                i = int(line.strip())
            except:
                i = -1

            try:
                del self.recorded_coordinates[i]
            except IndexError:
                print('*** Index out of bounds')
                return

            if not self.quiet:
                self.print_recorded_coordinates()
        else:
            print('*** No recorded coordinates')

    def do_d(self, line):
        """
        Alias of `delete'.
        """
        self.do_delete(line)

    def do_write(self, line):
        """
        Write recorded coodinates as an int32 numpy array of shape (N, 2) to
        FILE in format npy.

        Usage: w FILE
        Example 1 (write to "out.npy"): w out.npy
        Example 2 (write to "filename with space.npy"): w filename with space.npy
        """
        if self.recorded_coordinates:
            coordinates = np.stack(self.recorded_coordinates)
            try:
                np.save(line, coordinates)
            except BaseException as e:
                print('***', e)
        else:
            print('*** No recorded coordinates')

    def do_w(self, line):
        """
        Alias of `write'.
        """
        self.do_write(line)

    def do_boxwrite(self, line):
        """
        Assuming that the first coordinate is the upper left corner of a
        bounding box, and that the second coordinate the bottom right corner
        of the bounding box. Converts the second coordinate to (Wb, Hb), where
        `Wb' is the width of the box and `Hb` the height. Then write them to
        file FILE. This command won't work unless there's only two coordinates
        recorded. The command won't fail if the first coordinate is in fact
        the bottom right and the second the upper left.

        Usage: boxwrite FILE
        """
        if len(self.recorded_coordinates) == 2:
            ul, br = self.recorded_coordinates
            if (ul > br).all():
                ul, br = br, ul
            if (ul < br).all():
                wh = br - ul
                towrite = np.stack((ul, wh))
                try:
                    np.save(line, towrite)
                except BaseException as e:
                    print('***', e)
                else:
                    if not self.quiet:
                        print('Written bounding box:')
                        print(towrite)
            else:
                print('*** The two recorded coordinates must be the upper left '
                      'corner and the bottom right corner of the bounding box, '
                      'and the width/height of the bbox must be at least 1')
        else:
            print('*** There must be exactly two coordinates recorded')

    def do_bw(self, line):
        """
        Alias of `boxwrite'.
        """
        self.do_boxwrite(line)

    def do_quiet(self, line):
        """
        Suppress all non-error stdout for all commands but `pr'.

        Usage: quiet
        """
        self.quiet = True

    def do_noquiet(self, line):
        """
        Opposite of `quiet'.

        Usage: noquiet
        """
        self.quiet = False

    def do_load(self, line):
        """
        Load npy file as recorded coordinates. If there're already recorded
        coordinates, they will be overwritten.

        Usage: load FILE
        Example 1 (load from "out.npy"): load out.npy
        Example 2 (load from "filename with space.npy"): load filename with space.npy
        """
        try:
            coordinates = np.load(line)
        except BaseException as e:
            print('***', e)
            return

        self.recorded_coordinates = list(coordinates)
        if not self.quiet:
            self.print_recorded_coordinates()

    def do_bload(self, line):
        """
        Load npy file written by `boxwrite' as recorded coordinates. If there're
        already recorded coordinates, they will be overwritten.

        Usage: bload FILE
        Example 1 (load from "out.npy"): bload out.npy
        Example 2 (load from "filename with space.npy"): bload filename with space.npy
        """
        try:
            coordinates = np.load(line)
        except BaseException as e:
            print('***', e)
            return

        xy, wh = list(coordinates)
        xy2 = xy + wh
        self.recorded_coordinates = [xy, xy2]
        if not self.quiet:
            self.print_recorded_coordinates()

    def do_drawbox(self, line):
        """
        Assuming that the first coordinate is the upper left corner of a
        bounding box, and that the second coordinate the bottom right corner
        of the bounding box. Draws a red bounding box from the two coordinates.
        This command won't work unless there's only two coordinates recorded.
        The command won't fail if the first coordinate is in fact the bottom
        right and the second the upper left.

        Usage: drawbox
        """
        if len(self.recorded_coordinates) == 2:
            ul, br = self.recorded_coordinates
            if (ul > br).all():
                ul, br = br, ul
            if (ul < br).all():
                aimg = draw_bbox(ul, br, img)
                self.updage_canvas(aimg)
            else:
                print('*** The two recorded coordinates must be the upper left '
                      'corner and the bottom right corner of the bounding box, '
                      'and the width/height of the bbox must be at least 1')
        else:
            print('*** There must be exactly two coordinates recorded')

    def do_clear(self, line):
        """
        Clear all annotations, but remains recorded coordinates intact.

        Usage: clear
        """
        self.updage_canvas(img)

    def do_exit(self, line):
        plt.close('all')
        return True

    def do_q(self, line):
        """
        Alias of `exit'.
        """
        return self.do_exit(line)

    def preloop(self):
        plt.close('all')
        plt.figure()
        plt.ion()
        plt.show()
        self.updage_canvas(img)


if not args.coordinate:
    CmdApp().cmdloop()
