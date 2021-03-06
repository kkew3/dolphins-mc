"""
Common training utilities
"""
import contextlib
import os
import re
import logging
from datetime import datetime
import collections
import configparser
import importlib
import functools
import typing

import numpy as np
import torch
import torch.nn as nn

from loglib import loggername


def _l(*args):
    return logging.getLogger(loggername(__name__, *args))


_Predicate = typing.Callable[[typing.Any], bool]
_EBProgress = typing.Tuple[int, int]
StrKeyDict = typing.Dict[str, typing.Any]
IniDictLike = typing.Union[configparser.ConfigParser,
                           typing.Dict[str, StrKeyDict]]
T = typing.TypeVar('T')


def action_fired(fired: typing.Union[int, _Predicate]) -> _Predicate:
    """
    Returns a callable that returns a bool indicating whether an action should
    be performed given the progress of an ongoing task.

    :param fired: an int or the callable to be returned. If
           ``fired`` is an int, then ``fired`` will be initialized to
           ``lambda x: x % fired == 0``
    """
    logger = _l('action_fired')
    if isinstance(fired, int):
        if fired < 1:
            logger.warning('Expect `fired` at least 1 if int but got {}; '
                           'converted to 1'.format(fired))
            fired = 1

        def _fired(progress: int):
            return progress % fired == 0

        return _fired
    return fired


def fired_always(_) -> bool:
    """
    Always fire whatever the progress.
    """
    return True


def fired_batch(batch: int, progress: _EBProgress) -> bool:
    """
    Fire every other ``batch`` given progress in (epoch, batch) tuple.
    """
    return progress[1] % max(1, int(batch)) == 0


class CheckpointSaver:
    """
    Save checkpoint periodically.
    """

    def __init__(self, net: nn.Module, savedir: str,
                 checkpoint_tmpl: str = 'checkpoint_{0}.pth',
                 fired: typing.Union[int, _Predicate] = 10):
        """
        :param net: the network to save states; the network should already been
               in the target device
        :param savedir: the directory under which to write checkpoints; if not
               exists, it will be created automatically
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        :param checkpoint_tmpl: the checkpoint file template
               (by ``str.format``), should accept exactly one positional
               argument the same as that passed to ``fired``
        """
        if not isinstance(net, nn.Module):
            raise TypeError('Expected torch.nn.Module of `net` but got: {}'
                            .format(type(net)))

        savedir = os.path.normpath(savedir)
        os.makedirs(savedir, exist_ok=True)

        self.net = net
        self.fired = action_fired(fired)
        self.checkpoint_tmpl = checkpoint_tmpl
        self.savedir = savedir

    def __call__(self, progress) -> typing.Optional[str]:
        """
        Save checkpoint as needed.

        :param progress: see ``help(type(self).__init__)``
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            _logger = _l(self)
            name = self.checkpoint_tmpl.format(*progress)
            tofile = os.path.join(self.savedir, name)
            torch.save(self.net.state_dict(), tofile)
            _logger.debug('{} written at progress {}'
                          .format(tofile, progress))
            return tofile
        return None


def load_checkpoint(net: nn.Module, savedir: str, checkpoint_tmpl: str,
                    progress: tuple, map_location: str = 'cpu') -> None:
    """
    Load checkpoint from file.

    :param net: network to load checkpoint
    :param savedir: directory under which checkpoints are saved
    :param checkpoint_tmpl: basename template of the checkpoint files
    :param progress: which checkpoint file to load
    :param map_location: where to load the weights
    """
    logger = _l('load_checkpoint')
    basename = checkpoint_tmpl.format(*progress)
    fromfile = os.path.join(savedir, basename)
    state_dict = torch.load(fromfile, map_location=map_location)
    net.to(map_location)
    net.load_state_dict(state_dict)
    logger.debug('progress {} loaded from {}'
                 .format(progress, fromfile))


class StatSaver:
    """
    Save (scalar) statistics periodically as npz file.
    """

    def __init__(self, statdir: str,
                 statname_tmpl='stats_{0}.npz',
                 fired: typing.Union[int, _Predicate] = 10):
        """
        :param statdir: the directory under which to write statistics npz
               files; if not exists, it will be created automatically
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        :param statname_tmpl: the stat npz file basename template
               (by ``str.format``), should accept exactly one positional
               argument the same as that passed to ``fired``
        """
        statdir = os.path.normpath(statdir)
        os.makedirs(statdir, exist_ok=True)

        self.fired = action_fired(fired)
        self.statname_tmpl = statname_tmpl
        self.statdir = statdir

    def __call__(self, progress, **stat_dict):
        """
        Save statistics, which are wrapped into numpy arrays, as needed.

        :param progress: see ``help(type(self).__init__)``
        :param stat_dict: dict of floats or lists of floats, i.e. the
               non-cumulative statistics at the progress
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            _logger = _l(self)
            name = self.statname_tmpl.format(*progress)
            tofile = os.path.join(self.statdir, name)
            _stat_dict = {k: np.array(stat_dict[k]) for k in stat_dict}
            np.savez(tofile, **_stat_dict)
            _logger.debug('{} written at progress "{}"'
                          .format(tofile, progress))
            return tofile
        return None


class FieldChangedError(Exception):
    def __init__(self, original: typing.Sequence[str],
                 now: typing.Sequence[str]):
        super().__init__('Fields changed from {} to {}'
                         .format(original, now))


class CsvStatSaver:
    """
    Save scalar statistics as CSV files. To simplify design and
    implementation, the ``progress`` is assumed to be 2-tuples of integers
    ``(epoch, batch)``. The naming template (``str.format``) of the CSV files
    should contain one argument for the epoch. Within each file, the first
    column will be the batch id. The rest columns will be the scalar
    statistics at each batch.
    """

    def __init__(self, statdir: str,
                 statname_tmpl: str = 'stats_{0}.csv',
                 fired: typing.Union[int, _Predicate] = 10):
        """
        :param statdir: the directory under which to write statistics npz
               files; if not exists, it will be created automatically
        :param statname_tmpl: the base filename of the CSV file
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        """
        statdir = os.path.normpath(statdir)
        os.makedirs(statdir, exist_ok=True)

        self.fired = action_fired(fired)
        self.statdir = statdir
        self.statename_tmpl = statname_tmpl
        self.fields = None

    def __call__(self, progress, **stat_dict):
        """
        Save statistics.
        Once instantiated, the keys of ``stat_dict`` must not be changed,
        otherwise raising ``trainlib.FieldChangedError``

        :param progress: see ``help(type(self).__init__)``
        :param stat_dict: dict of floats i.e. the non-cumulative
               *scalar* statistics at the progress
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            _logger = _l(self)
            epoch, batch = progress
            name = self.statename_tmpl.format(epoch)
            tofile = os.path.join(self.statdir, name)
            if self.fields is None:
                self.fields = sorted(stat_dict)
            elif self.fields != sorted(stat_dict):
                raise FieldChangedError(self.fields, sorted(stat_dict))
            values = [stat_dict[k] for k in self.fields]
            try:
                open(tofile).close()
            except FileNotFoundError:
                with open(tofile, 'w') as outfile:
                    # write the CSV header
                    outfile.write(','.join([''] + self.fields) + '\n')
                    outfile.write(','.join(map(str, [batch] + values)) + '\n')
                    _logger.debug('{} written at progress "{}"'
                                  .format(tofile, progress))
            else:
                with open(tofile, 'a') as outfile:
                    outfile.write(','.join(map(str, [batch] + values)) + '\n')
                    _logger.debug('{} appended at progress "{}"'
                                  .format(tofile, progress))
            return tofile
        return None


def load_stat(statdir: str, statname_tmpl: str, progress: tuple,
              key: str = None) \
        -> typing.Union[typing.Dict[str, np.ndarray], np.ndarray]:
    """
    Load statistics dumped by ``StatSaver``.

    :param statdir: directory under which the statistics are saved
    :param statname_tmpl: stat file basename template
    :param progress: which stat file to load
    :param key: which field to load
    :return: the npz content dict if ``key`` is not specified, otherwise the
             corresponding field data
    """
    logger = _l('load_stat')
    basename = statname_tmpl.format(*progress)
    fromfile = os.path.join(statdir, basename)
    data = np.load(fromfile)
    logger.debug('progress {} loaded from "{}"'
                 .format(progress, fromfile))
    if key is None:
        data = {k: data[k] for k in data.keys()}
    else:
        data = data[key]
    return data


def freeze_model(model: nn.Module) -> typing.Dict[str, bool]:
    """
    Freeze the model parameters.

    :param model:
    :return: the original ``requires_grad`` attributes of the model parameters
    """
    origrg = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    return origrg


def melt_model(model: nn.Module, origrg: typing.Dict[str, bool],
               empty_after_melting=False) -> None:
    """
    Melt the model parameters into their original ``requires_grad`` states.

    :param model:
    :param origrg: dict mapping ``parameter_name`` to
           ``original_requires_grad``
    :param empty_after_melting: if ``True``, make ``origrg`` empty at the
           end of this function
    """
    for n, p in model.named_parameters():
        p.requires_grad_(origrg[n])
    if empty_after_melting:
        origrg.clear()


class TrainerConfig:
    """
    Attributes:

    +-----------------+-----------------------------------------------------+
    |    Attribute    |                     Description                     |
    +=================+=====================================================+
    | ``attrs``       | The key-value pairs under ``$ROOT_SECTION``.        |
    |                 | These key-value pairs means to be global or default |
    |                 | attributes.                                         |
    +-----------------+-----------------------------------------------------+
    | ``stage_attrs`` | The key-value pairs under section named from the    |
    |                 | name of each stage. These are stage-specific        |
    |                 | settings and override the default values from       |
    |                 | ``attrs``.                                          |
    +-----------------+-----------------------------------------------------+
    """
    ROOT_SECTION = '__root__'
    REGISTERED_FIRED_FUNCTIONS = {
        'fired_always': fired_always,
        'fired_batch': fired_batch,
    }

    def __init__(self, cfg: IniDictLike):
        self.attrs: typing.Dict[str, typing.Any] = self._parse_root(cfg)
        """the root attributes"""

        self.stage_attrs: typing.Dict[str, typing.Dict[str, typing.Any]] = {
            stage: self._parse_stage(self.attrs, stage, cfg)
            for stage in self.attrs['run_stages']
        }
        """the stage-specific attributes"""

    @staticmethod
    def __proc_or_default(sec: StrKeyDict, key: str,
                          f: typing.Callable[[str], T],
                          default: T) -> T:
        try:
            v = sec[key]
        except KeyError:
            return default
        else:
            return f(v)

    def _parse_root(self, cfg: IniDictLike):
        """
        Root section may or may not exists.
        """
        sec = cfg.get(self.ROOT_SECTION, {})
        attrs: typing.Dict[str, typing.Any] = {}

        attrs['run_stages'] = self.__proc_or_default(
            sec, 'run_stages', self._parse_identifier_list, ())
        if self.ROOT_SECTION in attrs['run_stages']:
            raise ValueError('Illegal stage ({}) in run_stages'
                             .format(self.ROOT_SECTION))

        attrs['stat_names'] = self.__proc_or_default(
            sec, 'stat_names', self._parse_identifier_list, ())

        attrs['savedir'] = self.__proc_or_default(
            sec, 'savedir', os.path.normpath, 'save')
        if os.path.isabs(attrs['savedir']):
            raise ValueError('__root__/savedir "{}" should be a relative '
                             'directory'.format(attrs['savedir']))
        attrs['fired'] = self.__proc_or_default(
            sec, 'fired', self._parse_fired, fired_always)
        return attrs

    def _parse_stage(self, root: typing.Dict[str, typing.Any],
                     stage: str, cfg: IniDictLike):
        """
        Parse stage-specific attributes.
        """
        sec = cfg.get(stage, {})
        attrs: typing.Dict[str, typing.Any] = {}
        attrs['statdir'] = self.__proc_or_default(
            sec, 'statdir', os.path.normpath, self.default_rel_statdir(stage))
        if os.path.isabs(attrs['statdir']):
            raise ValueError('{}/statdir "{}" should be a relative directory'
                             .format(stage, attrs['statdir']))
        attrs['fired_stat'] = self.__proc_or_default(
            sec, 'fired_stat', self._parse_fired, root['fired'])
        attrs['stat_names'] = self.__proc_or_default(
            sec, 'stat_names', self._parse_identifier_list, root['stat_names'])
        return attrs

    @staticmethod
    def default_rel_statdir(stage):
        name = 'stat'
        if stage != 'train':
            name = '_'.join((name, stage))
        return name

    @classmethod
    def _parse_fired(cls, fired: str) -> typing.Optional[typing.Callable]:
        """
        If empty, returns ``None``.
        """
        _fired_and_args = fired.lstrip().split(',')
        if len(_fired_and_args) == 1 and not _fired_and_args[0].strip():
            # this means that the key is specified as empty or white characters
            return None

        _fired = _fired_and_args[0].strip()
        _fired_args = _fired_and_args[1:]
        try:
            ffired = cls.REGISTERED_FIRED_FUNCTIONS[_fired]
            ffired = functools.partial(ffired, *_fired_args)
        except KeyError:
            # treat ``_fired`` as the qualname of a function,
            # i.e.: module_name.function_name
            mfired, ffired = _fired.rsplit('.', maxsplit=1)
            ffired = getattr(importlib.import_module(mfired), ffired)
            ffired = functools.partial(ffired, *_fired_args)
        return ffired

    @staticmethod
    def _parse_identifier_list(string: str) -> typing.Tuple[str, ...]:
        return tuple(filter(None, re.split(r'[ \t]*[,;] *|[ \t]+', string)))


class BasicTrainer:
    """
    Abstract class of a basic trainer that defines the general framework of
    training a network.  The class defines yet to be implemented callbacks
    intended to be overridden in subclasses. ``BasicTrainer`` is integrated
    with the following functions:

        - logging
        - dumping checkpoints to ``.pth`` files
        - dumping runtime statistics to ``.npz`` files
        - resume training halfway
        - evaluation after training

    The ``BasicTrainer`` is executed for ``max_epoch`` epochs.  For each
    epoch, it runs ``len(run_stages)`` stages of loops.  Dynamic hook methods
    are inserted to ensure flexibility. Pseudocode illustrating how it runs
    in general::

        .. code-block::

            # global-enter-hook
            for epoch in range(max_epoch):
                # epoch-enter-hook
                for stage in run_stages:
                    # stage-enter-hook
                    for batch in dataloader(stage):
                        # batch-enter-hook
                        ...
                        # batch-exit-hook
                    # stage-exit-hook
                # epoch-exit-hook
            # global-exit-hook

    ``ini`` configuration file
    ++++++++++++++++++++++++++

    How checkpoints and runtime statistics are saved is configured using an
    ``.ini`` configuration file.  The configuration file specified at current
    working directory will override the one under the same directory of the
    trainer class definition.  The ``ini`` file basename, default to
    ``trainer.ini``, may be changed at ``__init__``.  The ``ini`` file
    structure is shown in the table below.

    +--------------+----------------+-----------------------------------------+
    |    Section   |     Key        |              Description                |
    +==============+================+=========================================+
    | ``__root__`` | ``run_stages`` | - Comma-separated identifier list       |
    |              |                |   specifying the ``STAGE`` to run in    |
    |              |                |   each epoch.                           |
    |              |                | - Default to empty list.                |
    |              +----------------+-----------------------------------------+
    |              | ``stat_names`` | - Comma-separated identifier list of    |
    |              |                |   statistics names to be returned by    |
    |              |                |   each ``STAGE_only`` method; the       |
    |              |                |   attribute can be overridden by        |
    |              |                |   ``stat_names`` in each STAGE section. |
    |              |                | - Default to empty list.                |
    |              +----------------+-----------------------------------------+
    |              | ``savedir``    | - Directory to hold checkpoints in loop |
    |              |                |   ``train``; must be under              |
    |              |                |   ``$basedir``.                         |
    |              |                | - If it's an absolute path or starts    |
    |              |                |   with ``./``, it will be treated as    |
    |              |                |   is; otherwise, it will be joined      |
    |              |                |   automatically with ``$basedir`` to    |
    |              |                |   form a sub-directory of ``$basedir``. |
    |              |                | - Default to ``$basedir/save``          |
    |              +----------------+-----------------------------------------+
    |              | ``fired``      | - Comma-separated list of firing policy |
    |              |                |   name (or the full-qualified firing    |
    |              |                |   policy function name if the name is   |
    |              |                |   not registered at                     |
    |              |                |   ``trainlib.TrainerConfig``) and       |
    |              |                |   its arguments if any.                 |
    |              |                | - Default to "always firing"            |
    +--------------+----------------+-----------------------------------------+
    |   STAGE      |                | - ``$STAGE`` is the stage name          |
    |              +----------------+-----------------------------------------+
    |              | ``statdir``    | - Directory to hold statistics in loop  |
    |              |                |   ``$STAGE``; must be under             |
    |              |                |   ``basedir``.                          |
    |              |                | - If it's an absolute path or starts    |
    |              |                |   with ``./``, it will be treated as    |
    |              |                |   is; otherwise, it will be joined      |
    |              |                |   automatically with ``$basedir`` to    |
    |              |                |   form a sub-directory of ``$basedir``. |
    |              |                | - Default to ``$basedir/stat`` if       |
    |              |                |   ``$STAGE`` is ``train``, otherwise to |
    |              |                |   ``$basedir/stat_STAGE``.              |
    |              +----------------+-----------------------------------------+
    |              | ``fired_stat`` | - Same format as ``fired``, specifying  |
    |              |                |   the firing policy of statistics saver |
    |              |                |   of ``$STAGE`` loop.                   |
    |              |                | - Default to ``$fired``                 |
    |              +----------------+-----------------------------------------+
    |              | ``stat_names`` | - Same format as                        |
    |              |                |   ``__root__/stat_names``, but for each |
    |              |                |   STAGE specifically, overriding the    |
    |              |                |   global setting.                       |
    |              |                | - Default to ``$__root__/stat_names``.  |
    +--------------+----------------+-----------------------------------------+

    Dynamic methods requierd to define before ``run``
    +++++++++++++++++++++++++++++++++++++++++++++++++

    For simplicity in implementation while maintaining flexibility for
    subclassing, some methods to override are based on convention rather than
    structural inheritance.  The following table lists these methods.

    +-----+--------------------+----------------------------------------------+
    | Man |      Method        |                 Description                  |
    +=====+====================+==============================================+
    | T   | STAGE_once         | Minibatch in loop STAGE                      |
    +-----+--------------------+----------------------------------------------+
    | T   | get_STAGEloader    | Dataloader of (inputs,targets) in loop STAGE |
    +-----+--------------------+----------------------------------------------+
    | F   | before_batch_STAGE | Launched before STAGE_once                   |
    +-----+--------------------+----------------------------------------------+
    | F   | after_batch_STAGE  | Launched after STAGE_once                    |
    +-----+--------------------+----------------------------------------------+
    | F   | before_stage_STAGE | Launched before all batches of stage STAGE   |
    |     |                    | while after ``net.train()`` or               |
    |     |                    | ``net.eval()``                               |
    +-----+--------------------+----------------------------------------------+
    | F   | after_stage_STAGE  | Launched after all batches of stage STAGE    |
    +-----+--------------------+----------------------------------------------+

    where ``Man`` denotes "Mandatory".  Without implementing mandatory methods
    leads to ``NotImplementedError`` when called from ``run`` (the start entry
    of the execution).  This table shows the method signature (without
    ``self``).

    +--------------------+-----------------------------------+
    |      Method        |             Signature             |
    +====================+===================================+
    | STAGE_once         | ``(*args: T) -> stats: Tuple``    |
    +--------------------+-----------------------------------+
    | get_STAGEloader    | ``() -> Iterator[T]``             |
    +--------------------+-----------------------------------+
    | before_batch_STAGE | ``() -> None``                    |
    +--------------------+-----------------------------------+
    | after_batch_STAGE  | ``() -> None``                    |
    +--------------------+-----------------------------------+
    | before_stage_STAGE | ``() -> None``                    |
    +--------------------+-----------------------------------+
    | after_stage_STAGE  | ``() -> None``                    |
    +--------------------+-----------------------------------+

    Optional instance variables before ``run``
    ++++++++++++++++++++++++++++++++++++++++++

    These instance variables can be specified at compile time or at runtime
    before ``run`` is called:

        - ``basedir``: the base directory of ``savedir`` and all ``statdir``
          specified in the ``.ini`` file.

    Optional class variables
    ++++++++++++++++++++++++

    These class variables can be specified at compile time:

        - ``DEFAULT_CONFIGNAME``: the basename of the default configuration of
          the ``Trainer`` class lying under the same directory with the class
          definition file
        - ``__FILE__``: assign this variable ``__file__`` so that
          ``DEFAULT_CONFIGNAME`` will be searched for under the dirname of
          ``__FILE__``; ``AttributeError`` will be raised if ``__FILE__`` is
          not specified when ``DEFAULT_CONFIGNAME`` is
    """

    timestamp_format = '%Y%m%d%H%M'  # e.g. 201811051438
    """
    Used to name the default ``basedir``.
    """

    checkpoint_tmpl = 'checkpoint_{0}_{1}.pth'
    """
    Checkpoint pth file name template, accepting progress tuple ``(epoch_id,
    minibatch_id)``.
    """

    statname_tmpl = 'stats_{0}_{1}.npz'
    """
    Statistics npz file name template, accepting progress tuple ``(epoch_id,
    minibatch_id)``.
    """

    def __init__(self, net: nn.Module, max_epoch: int = 1,
                 device: str = 'cpu', progress: _EBProgress = None,
                 configname: str = 'trainer.ini', basedir: str = None):
        """
        :param net: the network to train
        :param max_epoch: maximum epoch to train, where an epoch is defined as
               a complete traversal of the underlying dataset
        :param device: where to train the network, choices:
               { cpu, cuda(:\\d+)? }
        :param progress: where to continue training, default to train from
               scratch. When ``progress`` is not ``None``, denote it as
               ``(E, B)``. The first checkpoint to be dumped by the trainer
               would be ``(E+1, 0)``, and will overwrite existing npy
               statistics and pth checkpoint files.
        :param configname: the ini filename used to configure the trainer.
               ``configname`` will override ``self.DEFAULT_CONFIGNAME`` if it
               exists. If none exists, raises ``FileNotFoundError``
        :param basedir: the base directory of all ``statdir`` and ``savedir``
               in the configuration. This attribute may or may not be
               specified here, but must be specified before the invocation of
               ``self.run``
        """
        self.device = device
        self.net = net
        self.max_epoch = max_epoch
        self.progress = progress
        self.configname = os.path.normpath(configname)
        ini = self.__collect_config()
        self.cfg = self.parse_config(ini)
        self.__log_parsed_config(self.cfg)
        with contextlib.suppress(ValueError):
            self.basedir = basedir
        self.__statsavers: typing.Dict[str, StatSaver] = {}
        self.__checkpointsavers: typing.Dict[str, CheckpointSaver] = {}

        self._trained_once = False
        """Used to mark the beginning of training"""

    def __collect_config(self) -> configparser.ConfigParser:
        configfiles = []
        if self.configname is not None:
            configfiles.append(self.configname)
        try:
            default_configname = getattr(type(self), 'DEFAULT_CONFIGNAME')
        except AttributeError:
            pass
        else:
            default_configfile = os.path.join(
                os.path.dirname(os.path.realpath(
                    getattr(type(self), '__FILE__'))),
                default_configname)
            if configfiles and not os.path.samefile(configfiles[0],
                                                    default_configfile):
                configfiles.append(default_configfile)

        cfg = configparser.ConfigParser()
        if configfiles:
            if not cfg.read(configfiles):
                raise FileNotFoundError(configfiles[0])
        else:
            raise RuntimeError('No configuration specified for {}'
                               .format(type(self).__name__))
        return cfg

    def __log_parsed_config(self, parsed_cfg: TrainerConfig) -> None:
        """
        This auxiliary method should be called after ``parse_config`` has
        returned.
        """
        tmpl_comps = []
        args = []
        for key, value in parsed_cfg.attrs.items():
            tmpl_comps.append('__root__/%s=%r')
            args.extend((key, value))
        for stage, _d in parsed_cfg.stage_attrs.items():
            for key, value in _d.items():
                tmpl_comps.append('%s/%s=%r')
                args.extend((stage, key, value))
        tmpl = ''.join(('Parsed trainer config: ', ', '.join(tmpl_comps)))
        logger = _l(self.__class__.__name__, self.__init__.__name__)
        logger.info(tmpl, *args)

    def parse_config(self, ini: configparser.ConfigParser) -> TrainerConfig:
        return TrainerConfig(ini)

    @property
    def basedir(self):
        try:
            return self._basedir
        except AttributeError:
            self._basedir = 'runs-{}'.format(datetime.today().strftime(
                type(self).timestamp_format))
            return self._basedir

    @basedir.setter
    def basedir(self, value: str):
        if not value:
            raise ValueError('basedir must not be empty')
        self._basedir = os.path.normpath(value)

    def init_dirs(self):
        """
        Initialize the directory settings according to the configuration
        file and all previous manual attribute settings. This function is
        intended to be called from ``setup``.
        """
        basedir = self.basedir
        self.cfg.attrs['savedir'] = \
            os.path.join(self.cfg.attrs['savedir'], basedir)
        for stage in self.cfg.stage_attrs:
            self.cfg.stage_attrs[stage]['statdir'] = \
                os.path.join(self.cfg.stage_attrs[stage]['statdir'], basedir)

    def prepare_net(self, ext_savedir: str = None) -> None:
        """
        Load checkpoint if ``progress`` is not ``None``, and move network to
        train to the specified device.

        :param ext_savedir: external savedir; if not set, use ``self.savedir``
        """
        savedir = ext_savedir or self.cfg.attrs['savedir']
        if self.progress is not None:
            load_checkpoint(self.net, savedir, type(self).checkpoint_tmpl,
                            self.progress, map_location=self.device)
        self.net.to(self.device)

    def __get_statsaver(self, stage):
        """Deferred instantiation of ``StatSaver``'s."""
        tmpl = type(self).statname_tmpl
        try:
            saver = self.__statsavers[stage]
        except KeyError:
            fired_stat = self.cfg.stage_attrs['fired_stat']
            if fired_stat is None:
                saver = None
            else:
                saver = StatSaver(self.cfg.stage_attrs['statdir'],
                                  statname_tmpl=tmpl, fired=fired_stat)
            self.__statsavers[stage] = saver
        return saver

    def __get_checkpointsaver(self):
        """Deferred instantiation of the ``CheckpointSaver``."""
        tmpl = type(self).checkpoint_tmpl
        try:
            saver = self.__checkpointsavers['train']
        except KeyError:
            fired = self.cfg.attrs['fired']
            if fired is None:
                saver = None
            else:
                saver = CheckpointSaver(self.net, self.cfg.attrs['savedir'],
                                        checkpoint_tmpl=tmpl, fired=fired)
            self.__checkpointsavers['train'] = saver
        return saver

    def __before_stage(self, stage):
        with contextlib.suppress(AttributeError):
            getattr(self, 'before_stage_{}'.format(stage))()

    def __after_stage(self, stage):
        with contextlib.suppress(AttributeError):
            getattr(self, 'after_stage_{}'.format(stage))()

    def __before_batch(self, stage):
        """Call ``before_batch_STAGE``."""
        with contextlib.suppress(AttributeError):
            getattr(self, 'before_batch_{}'.format(stage))()

    def __after_batch(self, stage):
        """Call ``after_batch_STAGE``."""
        with contextlib.suppress(AttributeError):
            getattr(self, 'after_batch_{}'.format(stage))()

    def __get_loader(self, stage):
        """Call ``get_STAGEloader``."""
        name = 'get_{}loader'.format(stage)
        try:
            f = getattr(self, name)
        except AttributeError as err:
            raise NotImplementedError(name) from err
        return f()

    def __once(self, stage, *args):
        """Call ``STAGE_once``."""
        name = '{}_once'.format(stage)
        try:
            f = getattr(self, name)
        except AttributeError as err:
            raise NotImplementedError(name) from err
        return f(*args)

    def before_epoch(self):
        """
        Callback before each epoch.
        """
        _l(self, self.before_epoch.__name__).debug('')

    def after_epoch(self):
        """
        Callback after each epoch.
        """
        _l(self, self.after_epoch.__name__).debug('')

    def setup(self):
        """
        Callback before ``run`` and after ``__init__``. Default to:

            - initializing checkpoing and statistics savers (``init_dirs``)
            - loading the checkpoint (``prepare_net``)
        """
        self.init_dirs()
        self.prepare_net()

    def teardown(self, error: typing.Optional[Exception]):
        """
        Callback before the return of ``run``, whether or not a successful
        return.

            - melt the network if necessary (``melt_net``)

        :param error: the cause of the return, or ``None`` if there's no
               error. Note that when not None, it's not necessarily of exactly
               type ``Exception`` -- might be exception subclass of it
        """
        # if self.freeze_net_when_necessary:
        #     self.melt_net()

    def run(self):
        logger = _l(self, self.run.__name__)
        logger.debug('Initializing')
        self.setup()

        # Since it's uneasy and not necessary to train from exact batch of the
        # loaded checkpoint, it will start training from the next epoch of the
        # checkpoint epoch
        if self.progress is not None:
            cpepoch, _ = self.progress
            epoch0 = cpepoch + 1
            if epoch0 >= self.max_epoch:
                logger.warning('No epoch left to run: current_epoch(1st)={} '
                               'max_epoch={}'.format(epoch0, self.max_epoch))
                self.teardown(None)
                return
        else:
            epoch0 = 0

        try:
            for epoch in range(epoch0, self.max_epoch):
                self.before_epoch()
                for stage in self.cfg.attrs['run_stages']:
                    logger.debug('Begin stage %s', stage)
                    if stage == 'train':
                        self._trained_once = True
                        self.net.train()
                        self.__before_stage(stage)
                        for batch, it in enumerate(self.__get_loader(stage)):
                            # `it` is `(inputs, targets)`, etc.
                            self.__before_batch(stage)
                            stats = self.__once(stage, *it)
                            assert isinstance(stats, tuple), str(type(stats))
                            self.__log_stats(epoch, stage, batch,
                                             logger, stats)
                            self.__dump_checkpoint(epoch, batch, logger)
                            self.__after_batch(stage)
                        self.__after_stage(stage)
                    else:
                        self.net.eval()
                        self.__before_stage(stage)
                        for batch, it in enumerate(self.__get_loader(stage)):
                            # `it` is `(inputs, targets), etc.`
                            self.__before_batch(stage)
                            stats = self.__once(stage, *it)
                            assert isinstance(stats, tuple), str(type(stats))
                            self.__log_stats(epoch, stage, batch,
                                             logger, stats)
                            self.__after_batch(stage)
                        self.__after_stage(stage)
                self.after_epoch()
            logger.info('Returns successfully')
            self.teardown(None)
        except Exception as err:
            logger.exception(err)
            self.teardown(err)
        except KeyboardInterrupt as err:
            logger.info(KeyboardInterrupt.__name__)
            self.teardown(err)
            raise

    def __organize_stats(self, stage: str, stats: tuple) -> StrKeyDict:
        """
        Log warning if number of stats mismatches the ``stat_names`` attribute
        of ``stage``. Returns empty dict if ``stats`` is empty.
        :param stage:
        :param stats:
        :return:
        """
        logger = _l(self.__organize_stats.__name__)
        stat_names = list(self.cfg.stage_attrs[stage]['stat_names'])
        n_stats = 0 if not stats else len(stats)
        err = ('stat_names for stage %s (%s) does not match the number of '
               'returned stats (%d)')
        if len(stat_names) < n_stats:
            logger.warning(err, stage, stat_names, n_stats)
            # conform to ``numpy.savez``'s convention
            stat_names.extend(['arr_{}'.format(i) for i in
                               range(len(stat_names), n_stats)])
        elif len(stat_names) > n_stats:
            logger.warning(err, stage, stat_names, n_stats)
            del stat_names[n_stats:]
        assert len(stat_names) == n_stats, str((len(stat_names), n_stats))

        return collections.OrderedDict(zip(stat_names, stats)) if stats else {}

    @staticmethod
    def __make_str_one_line(s: str, max_affices=(35, 15)) -> str:
        n_pfx, n_sfx = max_affices
        s = s.replace('\n', r'\n')
        if len(s) > sum(max_affices):
            s = ' ... '.join((s[:n_pfx], s[-n_sfx:]))
        return s

    def __repr_stats(self, organized_stats: StrKeyDict) -> str:
        """
        Represent stats in one lines.
        """
        stringified = []
        for k, v in organized_stats.items():
            str_ = repr if isinstance(v, str) else str
            k = repr(k)
            v = self.__make_str_one_line(str_(v))
            stringified.append((k, v))
        return ''.join((
            '[',
            ', '.join('({}, {})'.format(*x) for x in stringified),
            ']',
        ))

    def __log_stats(self, epoch: int, stage: str, batch: int,
                    logger: logging.Logger, stats: tuple) -> None:
        stats_to_log = self.__organize_stats(stage, stats)
        if stats_to_log:
            statsaver = self.__get_statsaver(stage)
            if statsaver is None:
                logger.debug('epoch%d/%s batch%d: StatSaver has been disabled',
                             epoch, stage, batch)
            else:
                statsaver((epoch, batch), **stats_to_log)
                stats_to_log_repr = self.__repr_stats(stats_to_log)
                logger.info('epoch%d/%s batch%d: %s', epoch, stage, batch,
                            stats_to_log_repr)
        else:
            logger.info('epoch%d/%s batch%d', epoch, stage, batch)

    def __dump_checkpoint(self, epoch: int, batch: int,
                          logger: logging.Logger):
        checkpointsaver = self.__get_checkpointsaver()
        if checkpointsaver is None:
            logger.debug('CheckpointSaver has been disabled')
        else:
            checkpointsaver((epoch, batch))


class FreezingTrainer(BasicTrainer):
    """
    Abstract trainer class that freeze the network parameters (but not
    disabling the gradients of the inputs) whenever ``stage`` is not "train".
    For details, refer to ``help(BasicTrainer)``.
    """

    def __init__(self, net: nn.Module, max_epoch: int = 1,
                 device: str = 'cpu', progress: _EBProgress = None,
                 configname: str = 'trainer.ini', basedir: str = None,
                 freeze_net_when_necessary=False):
        """
        :param net: the network to train
        :param max_epoch: maximum epoch to train, where an epoch is defined as
               a complete traversal of the underlying dataset
        :param device: where to train the network, choices:
               { cpu, cuda(:\\d+)? }
        :param progress: where to continue training, default to train from
               scratch. When ``progress`` is not ``None``, denote it as
               ``(E, B)``. The first checkpoint to be dumped by the trainer
               would be ``(E+1, 0)``, and will overwrite existing npy
               statistics and pth checkpoint files.
        :param configname: the ini file basename used to configure the trainer
        :param basedir: the base directory of all ``statdir`` and ``savedir``
               in the configuration. This attribute may or may not be
               specified here, but must be specified before the invocation of
               ``self.setup``
        :param freeze_net_when_necessary: if True, freeze the network
               parameters whenever ``stage`` is not 'train', and
               always freeze the network if there's no 'train' in
               ``self.run_stages``. Do not specify as ``True`` if in non-train
               stages the network weights are used for backpropagation. This
               option can be useful if the inputs requires gradient
               backpropagation, in which case ``torch.no_grad`` cannot be
               used
        """
        super().__init__(net, max_epoch=max_epoch, device=device,
                         progress=progress, configname=configname,
                         basedir=basedir)
        self.freeze_net_when_necessary = freeze_net_when_necessary
        self._origrg = {}
        """Used to freeze network when necessary"""
        self._frozen_always = False

    def freeze_net(self) -> None:
        self._origrg = freeze_model(self.net)

    def melt_net(self) -> None:
        melt_model(self.net, self._origrg, empty_after_melting=True)

    def before_stage_train(self):
        if self.freeze_net_when_necessary and not self._frozen_always:
            self.melt_net()

    def after_stage_train(self):
        if self.freeze_net_when_necessary and not self._frozen_always:
            self.freeze_net()

    def setup(self):
        """
        Callback before ``run`` and after ``__init__``. Default to:

            - initializing checkpoing and statistics savers (``init_dirs``)
            - loading the checkpoint (``prepare_net``)
            - freeze the network if necessary (``freeze_net``)
        """
        super().setup()
        run_stages = self.cfg.attrs['run_stages']
        if self.freeze_net_when_necessary and 'train' not in run_stages:
            self._frozen_always = True
        if self._frozen_always:
            self.freeze_net()


class BasicEvaluator(BasicTrainer):
    """
    Abstract base evaluator adapting the framework by ``BasicTrainer`` such
    that it's dedicated to evaluating an existing network checkpoint. It makes
    several tricky part of the original framework easier to use when applying
    to evaluation that goes over the dataset *once*.
    """

    def __init__(self, net: nn.Module, progress: _EBProgress,
                 basedir: str, device: str = 'cpu',
                 configname: str = 'trainer.ini'):
        """
        :param net: the network to evaluate
        :param progress: which to evaluate, must be of form (EPOCH, BATCH)
        :param basedir: the ``basedir`` of the original trainer
        :param device: where to evaluate
        :param configname: the ini file basename used to configure the
               original trainer
        """
        super().__init__(net, max_epoch=progress[0] + 2, device=device,
                         progress=progress, configname=configname)
        if not os.path.isdir(basedir):
            raise FileNotFoundError('basedir "{}" not found'.format(basedir))
        self.basedir = basedir

    def parse_config(self, ini: configparser.ConfigParser) -> TrainerConfig:
        cfg = super().parse_config(ini)
        if 'train' in cfg.attrs['run_stages']:
            raise ValueError('"train" should not exist in __root__/run_stages'
                             ' ({})'.format(cfg.attrs['run_stages']))
        return cfg

    def prepare_net(self, ext_savedir: str = None) -> None:
        savedir = ext_savedir or self.cfg.attrs['savedir']
        load_checkpoint(self.net, savedir, type(self).checkpoint_tmpl,
                        self.progress, map_location=self.device)
        self.net.to(self.device)
