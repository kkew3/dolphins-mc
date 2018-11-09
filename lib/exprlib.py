"""
Common utilities used in experiments.
"""

import os
import re
import json
import tempfile
import contextlib
from types import SimpleNamespace

import numpy as np


class ExperimentLauncher(object):
    """
    Launch experiment of which the results are to be stored in a single file.
    Returns the results directly if the experiment has been launched before
    with the same arguments; otherwise run the experiment and save the results.
    The experiment parameters are encoded in JSON string.

    Attributes:

        - result_filename: the result filename of the previous launch
    """

    def __init__(self, basedir, prefix='', suffix='', translate_kwargs=str):
        """
        :param basedir: the base directory to load/store results
        :param prefix: the prefix of the stored result and parameter JSON file
        :param suffix: the suffix of the stored result file; e.g. the file
               extension name
        :param translate_kwargs: a function that takes in the value of one
               keyword argument of the experiment and returns a JSON
               serializable object; a dictionary of keyword-function pairs is
               also accepted. When either ``translate_kwargs`` or the values
               of ``translate_kwargs`` is not callable, it's regarded as a
               function that always returns that value. Default to ``str``
        """
        self.basedir = os.path.normpath(basedir)
        self.prefix = prefix
        self.suffix = suffix
        self.translate_kwargs = translate_kwargs

        # the result filename of previous launch
        self.result_filename = None

    def load_result(self, filename: str) -> tuple:
        """
        Load experiment result from ``filename``.
        """
        raise NotImplementedError()

    def store_result(self, filename, *args):
        """
        Store anything returned by ``run`` to ``filename``, to be loaded by
        ``load_result`` afterwards.
        """
        raise NotImplementedError()

    def run(self, **kwargs) -> tuple:
        """
        Run the experiment with keyword arguments.
        """
        raise NotImplementedError()

    def encode_kwargs(self, kwdict: dict) -> dict:
        tr = self.translate_kwargs
        if not hasattr(tr, '__getitem__'):
            tr = {k: tr for k in kwdict}

        encdict = {}
        for k, v in kwdict.items():
            try:
                s = tr[k](v)
            except TypeError:
                s = tr[k]
            encdict[k] = s
        return encdict

    def search_history_params(self, enckwargs: dict):
        enckwargs = json.loads(json.dumps(enckwargs))
        found = None
        for filename in iter(os.path.join(self.basedir, x)
                             for x in os.listdir(self.basedir)
                             if x.endswith('.json')):
            try:
                with open(filename) as infile:
                    jo = json.load(infile)
            except (IOError, json.JSONDecodeError):
                pass
            else:
                if jo == enckwargs:
                    found = os.path.splitext(filename)[0]
                    self.result_filename = found
                    break
        if found:
            return self.load_result(found)

    def __call__(self, **kwargs):
        enckwargs = self.encode_kwargs(kwargs)
        result = self.search_history_params(enckwargs)
        if not result:
            new_result = self.run(**kwargs)
            with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                             dir=self.basedir,
                                             prefix=self.prefix,
                                             suffix=self.suffix) as outfile:
                filename = outfile.name
            self.store_result(filename, *new_result)
            self.result_filename = filename
            with open(filename + '.json', 'w') as outfile:
                json.dump(enckwargs, outfile)
            result = new_result
        return result


@contextlib.contextmanager
def fig_as_data(plt, fig, ax, with_alpha=False):
    plt.axis('off')
    ns = SimpleNamespace()
    try:
        yield ns
    except:
        raise
    else:
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer)
        if not with_alpha:
            data = data[...,:3]
        ns.data = data
    finally:
        plt.close()


def get_runid_from_file(_file_: str, return_prefix=False) -> str:
    r"""
    Extract runid from __file__, provided that __file__ is of format
    ``{something}.{runid}.py``, strictly speaking, of regex
    ``^[^\n\t\.]*\.([^\n\t\.]+)\.py$``.

    >>> get_runid_from_file
    :param _file_: the ``__file__`` constant
    :param return_prefix: if True, also returns the ``something`` part
    :return: the runid
    """
    pattern = r'^([^\n\t\.]*)\.([^\n\t\.]+)\.py$'
    matched = re.match(pattern, os.path.basename(_file_))
    if not matched:
        raise ValueError('"{}" not matching pattern \'{}\''
                         .format(_file_, pattern))
    if return_prefix:
        return matched.group(2), matched.group(1)
    else:
        return matched.group(2)
