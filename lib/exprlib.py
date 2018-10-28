"""
Common utilities used in experiments.
"""

import os
import json
import tempfile


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
