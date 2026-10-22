import contextvars
import logging
from typing import Callable, DefaultDict, List, Optional, Sequence, Type, TypeVar

from experimaestro import Config

logger = logging.getLogger("xpm_torch.context")
logger.setLevel(logging.INFO)

ContextType = TypeVar("ContextType", bound="Context")

_current_context: contextvars.ContextVar[Optional["Context"]] = contextvars.ContextVar(
    "current_context", default=None
)


class Hook(Config):
    """Base class for all hooks"""

    pass


HookType = TypeVar("HookType")


class InitializationHook(Hook):
    """Base class for hooks before/after initialization"""

    def after(self, context: "Context"):
        """Called after initialization"""
        pass

    def before(self, context: "Context"):
        """Called before initialization"""
        pass


class Context:
    """Generic computational context managing hooks and ambient scope.

    A ``Context`` maintains a registry of :class:`Hook` instances mapped by their
    type hierarchy and provides ambient context management via Python's :mod:`contextvars`.
    This avoids threading context objects down every layer of a call stack, allowing
    nested components to retrieve the active context via :meth:`Context.current`.

    Examples:
        Activating and accessing ambient context:

        >>> with my_context:
        ...     # Inside any nested function or module:
        ...     ctx = Context.current()
        ...     if ctx is not None:
        ...         ctx.call_hooks(MyHook, hook_method, *args)
    """

    hooksmap: dict[Type, List[Hook]]
    """Map of hooks"""

    def __init__(self, hooks: Optional[Sequence[Hook]] = None):
        self.hooksmap = DefaultDict(lambda: [])
        self._tokens: List[contextvars.Token] = []
        if hooks:
            for hook in hooks:
                self.add_hook(hook)

    def hooks(self, cls: Type[HookType]) -> List[HookType]:
        """Returns all the hooks"""
        return self.hooksmap.get(cls, [])  # type: ignore

    def call_hooks(self, cls: Type, method: Callable, *args, **kwargs):
        for hook in self.hooks(cls):
            method(hook, *args, **kwargs)

    def add_hook(self, hook):
        for cls in hook.__class__.__mro__:
            self.hooksmap[cls].append(hook)

    def __enter__(self):
        token = _current_context.set(self)
        try:
            self._tokens.append(token)
        except AttributeError:
            self._tokens = [token]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        tokens = getattr(self, "_tokens", None)
        if not tokens:
            logger.warning(
                f"Context.__exit__ called for {self} without matching __enter__"
            )
            return

        token = tokens.pop()
        _current_context.reset(token)

    @classmethod
    def current(cls: Type[ContextType]) -> Optional[ContextType]:
        """Returns the currently active Context (or subclass), or None if not set."""
        ctx = _current_context.get()
        if isinstance(ctx, cls):
            return ctx
        return None
