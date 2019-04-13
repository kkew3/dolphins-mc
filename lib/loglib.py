"""
Utils in logging.

Each logger name is partitioned into several parts. For a function, it's
partitioned into ``module`` and ``name``; for a class method, it's partitioned
into ``module``, ``class`` and ``name``. For simplicity, the level of partition
can be denoted respectively as one letter: ``m`` for ``module``, ``c`` for
``class`` and ``n`` for ``name``.


WithLoggerDecorator (four derivatives)
--------------------------------------

    - ``function_with_logger``
    - ``function_with_logger_upto``
    - ``method_with_logger``
    - ``method_with_logger_upto``

Mechanism: inject proper logger name into the keyword argument list of
the wrapped function, such that the keyword argument named
``name`` (default to ``__logger__``) is assigned the logger name
if not otherwise specified by the function caller.

Usage example (assuming the example functions are in file ``FILE.py``)::

    .. code-block::

        @function_with_logger
        def greeting(__logger__=None):
            """'"""'"""
            No parameters for this function.

            :return: 'hello'
            """'"""'"""
            __logger__.info('Called')
            return 'hello'

        @function_with_logger_upto('m')
        def greeting_again(n: int, __logger__=None):
            """'"""'"""
            :param n: number of times to say 'hello'
            :return: 'hello' repeted ``n`` times
            """'"""'"""
            __logger__.info('Called')
            return ' '.join(['hello'] * n)

In another file MAIN.py::

    .. code-block::

        import logging
        import FILE

        logging.basicConfig(level=logging.INFO)
        print(FILE.greeting())
        print(FILE.greeting_again(4))

Running ``main.py`` yields (order may vary)::

    FILE.greeting:INFO:Called
    hello
    FILE:INFO:Called
    hello hello hello hello


Similar usage applies to ``method_with_logger`` and ``method_with_logger_upto``
except that they are designed for class methods. Be sure to apply them before
``@classmethod`` or ``@staticmethod`` if any.
"""

__all__ = [
    'loggername',
    'function_with_logger',
    'function_with_logger_upto',
    'method_with_logger',
    'method_with_logger_upto',
]

import logging
import inspect
import functools
import typing

SIMPLE_LOGLEVEL = {
    'm': 'module',
    'c': 'class',
    'n': 'name',
}

LOGLEVELS_F = 'module', 'name'
LOGLEVELS_M = 'module', 'class', 'name'

LOGGER_PARAM_NAME = '__logger__'


def loggername(module_name: str, self_or_function_name: typing.Any,
               method_name: typing.Optional[str] = None) -> str:
    """
    Returns logger name. Usage::

        .. code-block::

            loggername(__name__)
            loggername(__name__, self)
            loggername(__name__, 'function_name')
            loggername(__name__, self, 'method_name')

    :param module_name: as returned by ``__name__``
    :param self_or_function_name: either the function name (str) or the
           ``self`` object of the method to log
    :param method_name: if not ``None``, the name of the method to log
    :return: the logger **name**
    """
    tokens = [module_name]
    if isinstance(self_or_function_name, str):
        tokens.append(self_or_function_name)
    else:
        tokens.append(type(self_or_function_name).__name__)
    if method_name:
        tokens.append(method_name)
    return '.'.join(tokens)


class WithLoggerDecorator:
    def __init__(self, levels, name: str = LOGGER_PARAM_NAME,
                 up_to_level: typing.Optional[str] = None):
        """
        :param name: the keyword argument to inject
        :param levels: either ``LOGLEVELS_F`` (for function decorator) or
               ``LOGLEVELS_M`` (for method decorator)
        :param up_to_level: if not ``None``, should be one of {'m', 'n',
               'module', 'name'} if ``levels`` is ``LOGLEVELS_F``, or one of
               {'m', 'c', 'n', 'module', 'class', 'name'} otherwise
        """
        self.levels = levels
        if up_to_level:
            up_to_level = SIMPLE_LOGLEVEL.get(up_to_level, up_to_level)
            self.up_to_level = LOGLEVELS_F.index(up_to_level)
        else:
            self.up_to_level = None
        self.logger = logging.getLogger(loggername(__name__, self))
        self.param_name = name

    def __call__(self, f):
        err = '%s does not have "%s" as keyword argument; skipped'
        parameters = inspect.signature(f).parameters
        try:
            par = parameters[self.param_name]
        except KeyError:
            for par in parameters.values():
                if par.kind == par.VAR_KEYWORD:
                    self.logger.debug('%s does not have "%s" as keyword '
                                      'argument but it has **kwargs; '
                                      'proceeded',
                                      f.__name__, self.param_name)
                    break
            else:
                self.logger.debug(err, f.__name__, self.param_name)
                return f
        else:
            if not ((par.kind == par.POSITIONAL_OR_KEYWORD
                     and par.default is par.empty)
                    or (par.kind == par.KEYWORD_ONLY)):
                self.logger.debug(err, f.__name__, self.param_name)
        logger = logging.getLogger('.'.join(self.levels[:self.up_to_level]))

        def wrapper(*args, **kwargs):
            if LOGGER_PARAM_NAME not in kwargs:
                kwargs[LOGGER_PARAM_NAME] = logger
            else:
                self.logger.debug('%s already in %s',
                                  LOGGER_PARAM_NAME, f.__name__)
            return f(*args, **kwargs)

        functools.update_wrapper(wrapper, f)
        return wrapper


function_with_logger = WithLoggerDecorator(LOGLEVELS_F)
function_with_logger_upto = functools.partial(WithLoggerDecorator, LOGLEVELS_F)
method_with_logger = WithLoggerDecorator(LOGLEVELS_M)
method_with_logger_upto = functools.partial(WithLoggerDecorator, LOGLEVELS_M)
