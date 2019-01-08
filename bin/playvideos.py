#!/usr/bin/env python
import logging
import argparse
import math
import os
import sys
import collections
import re
from typing import List, Iterator, Callable, Sequence

import numpy as np
import cv2
from tqdm import tqdm

import utils
import exprlib

try:
    from utils import frameiter
except ImportError:
    def frameiter(cap: cv2.VideoCapture, rgb: bool = True):
        while cap.isOpened():
            s, f = cap.read()
            if not s:
                break
            if len(f.shape) == 3 and f.shape[2] == 2 and rgb:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            yield f

__description__ = '''
Play videos synchronously with automatic window placement.

Keymaps
-------

    b) freeze the videos;
    c) continue playing the videos;
    g) show progress on console;
    h) show help on console;
    l) go to the earliest frame within the rewind limit;
    n) go to next frame;
    p) go to previous frame already played;
    r) go to the latest frame.

Example
-------

Play `video1' and `video1' side-by-side:

    python playvideos.py -f video1 -l 0 0 -f video2 -l 0 1;

Arrange all AVI videos in two rows:

    find . -name '*.avi' -print | python playvideos.py -f- -L 2 x

Custom frame preprocessing routine
----------------------------------

Routines can be injected at runtime by giving python files defining them.
The routines will be chained such that the result from the ith routine
will be fed as input into the (i+1)th routine. Each file defines one and
only one routine.

Each file defining a routine must contain a function named `frame_processor`
that expects no argument and returns a callable object, denoted as `fp`. The
callable object should behave the same as the following function signature:

    def fp(cl_frames: List[Tuple[Tuple[int, int], Optional[numpy.ndarray]]]) \\
            -> List[Tuple[Tuple[int, int], Optional[numpy.ndarray]]]:
        ...

`cl_frames' is a list of tuples (cell_location, frame).
For example,

    [((0,0),frame1), ((0,1),frame2), ((1,0),frame3)]

means frame1 at upper left, frame2 at upper right, frame3 at lower left,
nothing (blank) at lower right. If `frame' is `None', then the cell
located at `cell_location' is empty. For example,

    [((0,0),frame1), ((0,1),frame2), ((1,0),frame3)]

is equivalent to

    [((0,0),frame1), ((0,1),frame2), ((1,0),frame3), ((1,1),None)]

The returned value is of the same format as the input list, but does not
necessarily maintain the same length. For example, given the input list

    [((0,0),frame1), ((0,1),frame2)]

the returned list can be

    [((0,0),frame1'), ((0,1),frame2'), ((1,0),frame1), ((1,1),frame2)]


                      == End of description ==
'''

__description__ = __description__.strip()


def identity_frame_processor(_):
    return _


class VideoPlayer(object):
    """
    A ``VideoPlayer`` play one or more videos simultaneously in cells
    arranged in a grid. The location of each cell is specified by coordinate
    (x,y), 0-indexed, with row-id as x and column-id as y.
    """

    def __init__(self, *sources, colormode='gray', fps=6.0,
                 margin=1, zip_policy='longest', rewind_limit=100,
                 frame_processors: Sequence[Callable] = ()) -> None:
        """
        :param sources: tuple(s) of ``(cell_location, frames)``, where
               ``cell_location`` is a int-tuple (x,y) indicating its cell
               location, and ``frames`` is an iterable of frames
        :param fps: default to 6 frames/sec
        :param colormode: either 'gray' or 'rgb'
        :param margin: width of row/column margin in pixel
        :param zip_policy: either 'longest' or 'shortest'
        :param rewind_limit: maximum number of frames to rewind
        :param frame_processor: a callable object that accepts a list of
               tuples ``(cell_location, Optional[frame])`` and returns a list
               of tuples of the same format, where ``None`` frame indicates
               empty cell *throughout* the video (due to layout)
        """
        self.fps = fps
        self.colormode = colormode
        self.margin = margin
        self.zip_policy = zip_policy
        if not frame_processors:
            frame_processor = identity_frame_processor
        else:
            frame_processor = utils.fcompose(frame_processors)
        self.frame_processor = frame_processor

        if len(set(x[0] for x in sources)) < len(sources):
            raise ValueError('cell location must be unique for each cell')
        grid_shape = (1 + max(s[0][0] for s in sources),
                      1 + max(s[0][1] for s in sources))
        grid = [[None for _ in range(grid_shape[1])]
                for _ in range(grid_shape[0])]
        for s in sources:
            cl, frames = s
            grid[cl[0]][cl[1]] = iter(frames)
        self.grid = grid  # type: List[List[Iterator[np.ndarray]]]
        self.grid_shape = grid_shape

        self.cache = collections.deque(maxlen=max(1, rewind_limit))

        # switches
        self.paused = False
        self.quiet = False

        self.cache_pointer = 0
        self.frame_counter = 0
        self.eof = False

    def _next_frames(self):
        frames = []
        succeed = False
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                try:
                    f = next(self.grid[i][j])
                except StopIteration:
                    if self.zip_policy == 'longest':
                        frames.append(((i, j), None))
                    else:
                        raise
                except TypeError:
                    pass
                else:
                    if len(f.shape) == 2 and self.colormode == 'rgb':
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    elif (len(f.shape) == 3 and f.shape[2] == 3
                          and self.colormode == 'gray'):
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    frames.append(((i, j), f))
                    succeed = True
        if not succeed:
            raise StopIteration
        frames = self.frame_processor(frames)
        return frames

    def _render_new(self):
        frames = self._next_frames()
        try:
            canvas = self._render_default()
        except AttributeError:
            self.grid_shape_rt = tuple(1 + np.max([cl for cl, _ in frames], axis=0))
            gp_shape = np.zeros((2,) + self.grid_shape_rt, dtype=np.int64)
            for (i, j), f in frames:
                gp_shape[:, i, j] = (0, 0) if f is None else f.shape[:2]
            row_heights = np.cumsum(np.max(gp_shape[0], axis=1) + self.margin)
            col_widths = np.cumsum(np.max(gp_shape[1], axis=0) + self.margin)
            canvas_height = row_heights[-1] - self.margin
            canvas_width = col_widths[-1] - self.margin
            self._canvas_shape = canvas_height, canvas_width
            self._ll_coors = (row_heights - self.margin,
                              col_widths - self.margin)
            canvas = self._render_default()
        for (i, j), f in frames:
            if f is not None:
                fh, fw = f.shape[:2]
                ful = self._ll_coors[0][i] - fh, self._ll_coors[1][j] - fw
                canvas[ful[0]:ful[0] + fh, ful[1]:ful[1] + fw] = f
        self.cache.append((self.frame_counter, canvas))
        self.frame_counter += 1
        return canvas

    def _render_old(self) -> np.ndarray:
        """
        Render the next frame by iterating along ``self.cache``. When
        exhausting the cache, raises StopIteration.

        :return: the next frame of dtype ``numpy.uint8``
        """
        canvas = self.cache[self.cache_pointer][1]
        self.cache_pointer += 1
        return canvas

    def _render_default(self) -> np.ndarray:
        """
        Render blank screen.

        :return: blank screen
        """
        if self.colormode == 'rgb':
            canvas = np.zeros(self._canvas_shape + (3,), dtype=np.uint8)
        else:
            canvas = np.zeros(self._canvas_shape, dtype=np.uint8)
        return canvas

    def render(self) -> str:
        """
        Render frame and get keyboard interaction.

        :return: the key pressed
        """
        if self.paused and len(self.cache):
            self.cache_pointer -= 1
        try:
            if self.cache_pointer:
                canvas = self._render_old()
            else:
                canvas = self._render_new()
        except StopIteration:
            self.eof = True
            canvas = self._render_default()
        cv2.imshow(self.window_name, canvas)
        return chr(cv2.waitKey(int(np.round(1000 / self.fps))) & 0xFF)

    def skip(self):
        try:
            _ = self._render_new()
        except StopIteration:
            self.eof = True

    def pause(self):
        self.paused = True

    def prevf(self):
        if not self.paused and not self.quiet:
            print('Cannot go to previous frame unless paused')
        else:
            self.cache_pointer = max(1 - len(self.cache),
                                     self.cache_pointer - 1)

    def nextf(self):
        if not self.paused and not self.quiet:
            print('Cannot go to next frame unless paused')
        else:
            self.cache_pointer = min(1, self.cache_pointer + 1)

    def pprevf(self):
        if not self.paused and not self.quiet:
            print('Cannot go to the earliest frame unless paused')
        else:
            if len(self.cache):
                self.cache_pointer = 1 - len(self.cache)

    def nnextf(self):
        if not self.paused and not self.quiet:
            print('Cannot go to the latest frame played unless paused')
        else:
            if len(self.cache):
                self.cache_pointer = -1

    def cont(self):
        self.paused = False

    def do_nothing(self):
        # intended to be empty
        pass

    def print_progress(self):
        print(self.frame_id, sep='/')

    @property
    def frame_id(self):
        fid = self.frame_counter - 1
        if self.cache_pointer:
            fid += self.cache_pointer
        return fid

    @property
    def window_name(self):
        return 'video {}x{}'.format(*self.grid_shape_rt)


def make_parser():
    parser = argparse.ArgumentParser(
            description=__description__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--video-file', dest='videos',
                        metavar='VIDEO', action='append',
                        help='the video(s) to play at sync; or `-\' to read '
                             'video filenames from stdin, one per line')
    l_group = parser.add_argument_group('Layout specification')
    l_group.add_argument('-L', '--grid-shape', metavar=('NROWS', 'NCOLS'),
                         nargs=2, dest='grid_shape', default=(1, 1),
                         help='the grid shape, default to 1 1, where at '
                              'least one of NROWS '
                              'and NCOLS should be a positive integer; the '
                              'one that\'s not a positive integer will be '
                              'regarded as undefined dimension; this option '
                              'will be ignored unless `-l X Y\' is not '
                              'specified')
    l_group.add_argument('-l', '--location', nargs=2,
                         metavar=('X', 'Y'), dest='locations',
                         action='append',
                         help='the ith location corresponds to the ith VIDEO')

    v_group = parser.add_argument_group('Video specification')
    v_group.add_argument('--fps', type=float, default=6.0,
                         help='the frame-per-second; default to %(default)s')
    v_group.add_argument('-c', '--color',
                         choices=['rgb', 'gray'], default='gray',
                         help='rgb video or gray video; '
                              'default to %(default)s')

    fp_group = parser.add_argument_group('Frame processors')
    fp_group.add_argument('-r', '--routine', action='append', metavar='FILE',
                          help='file that defines a custom frame processor')

    b_group = parser.add_argument_group('Runtime behavior')
    b_group.add_argument('-b', '--freeze-once-start', dest='freeze_once_start',
                         action='store_true',
                         help='to freeze the videos once started')
    b_group.add_argument('-n', '--cache-size', type=int, default=100,
                         dest='cache',
                         help='how many number of frames the video player is '
                              'able to rewind; should be no less than 1; '
                              'default to %(default)s. '
                              'There is a limit because the '
                              'program cache last CACHE frames on the run. The'
                              ' reason why ffmpeg backend is not used because '
                              'it is not reliable (see '
                              'https://github.com/opencv/opencv/issues/9053).')
    b_group.add_argument('-g', '--goto', dest='start_frame',
                         metavar='FRAME_ID', default=0, type=int,
                         help='Start the videos at frame FRAME_ID')
    b_group.add_argument('-q', '--quiet', action='store_true')

    return parser


keymap = {
    'b': 'pause',
    'c': 'cont',
    'g': 'print_progress',
    'l': 'pprevf',
    'n': 'nextf',
    'p': 'prevf',
    'r': 'nnextf',
}


def loop(player: VideoPlayer, paused=False, startat=0):
    if startat < 500 or player.quiet:
        _it = range(startat)
    else:
        _it = tqdm(range(startat), ascii=True, unit='fm')
    for _ in _it:
        player.skip()
    if paused:
        player.pause()
    key = player.render()
    while key != 'q':
        if key == 'h':
            print('''
    b) freeze the videos;
    c) continue playing the videos;
    g) show progress on console;
    h) show help on console;
    l) go to the earliest frame within the rewind limit;
    n) go to next frame;
    p) go to previous frame;
    r) go to the latest frame already played.'''.strip())
        else:
            reaction = keymap.get(key, 'do_nothing')
            getattr(player, reaction)()
        key = player.render()


if __name__ == '__main__':
    args = make_parser().parse_args()
    if not args.videos:
        if not args.quiet:
            print('Nothing to play')
        sys.exit(0)

    caps = []
    if args.videos[0] == '-':
        if not sys.stdin.isatty():
            args.videos = list(map(os.path.normpath,
                                   map(str.strip, sys.stdin.readlines())))
            if args.videos is None or not len(args.videos):
                if not args.quiet:
                    print('Nothing to play')
                sys.exit(0)
        else:
            if not args.quiet:
                print('Expecting input from stdin', file=sys.stderr)
            sys.exit(1)
    if args.locations:
        if len(args.locations) < len(args.videos):
            if not args.quiet:
                print('Locations are not specified for some videos;'
                      ' aborted', file=sys.stderr)
            sys.exit(1)
        for l, v in zip(args.locations, args.videos):
            caps.append((l, cv2.VideoCapture(v)))
    else:
        nrows, ncols = args.grid_shape
        try:
            nrows = int(nrows)
        except (TypeError, ValueError):
            ncols = int(ncols)
            nrows = math.ceil(len(args.videos) / ncols)
        else:
            try:
                ncols = int(ncols)
            except (TypeError, ValueError):
                ncols = math.ceil(len(args.videos) / nrows)
        for i, v in enumerate(args.videos):
            caps.append(((i // ncols, i % ncols), cv2.VideoCapture(v)))
    iters = [(tuple(map(int, l)), frameiter(cap, rgb=False))
             for l, cap in caps]
    fproc = []
    if args.routine:
        for routine_file in args.routine:
            with open(routine_file) as infile:
                routine_src = infile.read()
            routine_code = compile(routine_src, routine_file, 'exec')
            _ns = {}
            exec(routine_code, _ns)
            try:
                fproc.append(_ns['frame_processor']())
            except (KeyError, TypeError):
                print('Expecting function `frame_processor` in "{}" that '
                      'returns the callable frame processor'
                      .format(routine_file), file=sys.stderr)
                raise

    player = VideoPlayer(*iters, colormode=args.color, fps=args.fps,
                         rewind_limit=args.cache, frame_processors=fproc)
    player.quiet = args.quiet
    try:
        loop(player, paused=args.freeze_once_start,
             startat=1 + args.start_frame)
    finally:
        for _, cap in caps:
            cap.release()
