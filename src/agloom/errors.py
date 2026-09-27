"""Public error hierarchy for Agloom."""


class AgloomError(Exception):
    """Base class for framework errors."""


class ConfigurationError(AgloomError, ValueError):
    """The public agent configuration is invalid."""


class TopologyError(AgloomError):
    """A topology could not be constructed or executed."""


class TopologySelectionError(TopologyError):
    """No valid execution choice could be made."""


class CapabilityError(AgloomError):
    """A capability operation failed."""


class CapabilityResolutionError(CapabilityError):
    """A capability is unavailable or could not be resolved."""


class RecursionError(AgloomError):
    """Recursive execution is disabled or invalid."""


class RecursionLimitError(RecursionError):
    """A recursive execution budget was exceeded."""


class CompilationError(AgloomError):
    """The architecture specification could not be compiled."""


class RuntimeError(AgloomError):
    """The harness runtime failed."""


class LifecycleError(RuntimeError):
    """An invalid lifecycle transition was requested."""


class ExecutionError(RuntimeError):
    """An execution failed."""


class CancellationError(RuntimeError):
    """Execution was cooperatively cancelled."""
