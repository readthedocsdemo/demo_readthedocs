import abc
import logging
import sys
import time
import math
import json


@dataclass
class Metric:
    """Class for representing metric produced during model testing"""
    name: str
    value: float

    def is_nan(self) -> bool:
        if self.value is None:
            return False
        return math.isnan(self.value)

    def is_inf(self) -> Union[bool, str]:
        if self.value is None:
            return False

        if not math.isinf(self.value):
            return False

        return {-math.inf: '-', math.inf: '+'}[self.value]
        

class CommandError(Exception):
    """
    Exception class indicating an error while executing a wqpt command.

    If this error is raised during the executing of command
    it will be caught and turned into a nicely-printed error
    message to the appropriate output stream.
    """


class BaseCommand:
    def add_arguments(self, parser):
        """Implement in subclass"""

    def register_command(self, parser):
        name = getattr(self, 'name', None)
        if not name:
            raise AttributeError('attribute name is required for command.')

        cls_help = getattr(self, 'help', None)
        if not cls_help:
            raise AttributeError('attribute help is required for command.')

        p = parser.add_parser(name, help=cls_help)
        self.add_arguments(p)
        p.set_defaults(func=self._process_args)

        return p

    def _process_args(self, args):
        args_ = vars(args)
        args_.pop('func')
        self.args = args_
        self.invoke(**args_)

    def set_output(self, output):
        self._output = output

    def get_output(self):
        return self._output

    @abc.abstractmethod
    def __call__(self):
        """To be implemented in concrete subclasses."""

    def invoke(self, *args, **kwargs):
        json_output = kwargs.get('json', False)

        exp = None
        with redirect_stdout(None if json_output else sys.stdout):
            try:
                self(*args, **kwargs)
            except SystemExit as e:
                exp = e

        if json_output:
            print(json.dumps(self._output, indent=4))

        if exp:
            raise exp

    @property
    def logger(self):
        if not self._logger:
            self._logger = logging.getLogger(str(type(self)))
        return self._logger

    @property
    def catalog(self):
        """Returns CatalogClient instance"""
        if not self._catalog:
            self._catalog = self._catalog_factory()

        return self._catalog


class ResultRegistrator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_metrics(self):
        """Returns metric or list of metrics"""


class BaseStatistic(ResultRegistrator):
    def __init__(self, y_true=None, y_pred=None):
        super().__init__()

        self._y_true = [] if y_true is None else y_true
        self._y_pred = [] if y_pred is None else y_pred

    def register(self, predicted, actual, args):
        if is_invalid_statistic_value(actual) or is_invalid_statistic_value(predicted):
            return

        self._y_true.extend(actual)
        self._y_pred.extend(predicted)


class StepBase:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unknown step {!r}'.format(self))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @abc.abstractmethod
    def execute(self, tester):
        """Run step"""


class BaseCodeGenerator:
    def __init__(self, wow, file=sys.stdout):
        self.wow = wow
        self._file = file

    @property
    def wow(self):
        return self.wow

    @abc.abstractmethod
    def prelude(self):
        """Prelude of generated script"""

    @abc.abstractmethod
    def postlude(self):
        """Postlude of generated script"""

    @abc.abstractmethod
    def set_state(self, command):
        """Set state call"""

    @abc.abstractmethod
    def fit(self, command):
        """Fit call"""

    @abc.abstractmethod
    def predict(self, command):
        """Predict call"""

    def generate(self, command):
        gen = getattr(self, command.cmd)
        output = gen(command)
        print(output, file=self._file)


class BaseLegacyWOWValidator:
    @abc.abstractmethod
    def validate(self, 
    ):
        """Validate the wow"""
        """Perform wow validation checks."""


@dataclass
class FailedWOWValidation:
    reasons: List[str]

    def __bool__(self):
        return False


class BaseWOWValidator:
    @abc.abstractmethod
    def validate(self, wow):
        """Perform file validation checks"""


class BaseMigration:
    @abc.abstractmethod
    def migrate(self):
        """Must be implemented in subclass"""

    @abc.abstractmethod
    def revert(self):
        """Must be implemented in subclass"""

    def is_mac(self):
        return sys.platform.lower() == 'darwin'

    def is_win(self):
        os_platform = sys.platform.lower()
        return os_platform.startswith('win') or os_platform in ['cygwin', 'msys']

    def is_linux(self):
        return sys.platform.lower().startswith('linux')
