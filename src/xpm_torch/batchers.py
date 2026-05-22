from abc import abstractmethod
from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Callable,
    Iterator,
    TypeVar,
    Union,
    Sequence,
)
import logging
import numpy as np
import torch
from experimaestro import Config, Param, field
from pytorch_lightning.utilities.memory import garbage_collection_cuda

logger = logging.getLogger("xpm_torch.batchers")

RT = TypeVar("RT")
T = TypeVar("T")
ARGS = TypeVar("ARGS")
KWARGS = TypeVar("KWARGS")


class RecoverableOOMError(Exception):
    """Exception raised when a OOM occurs and the batcher
    is taking this into account for the next round (i.e. we
    can reprocess the batch with a higher probability of not
    having an OOM)
    """

    pass


class IterativeProcessor(Protocol, Generic[T, RT, ARGS, KWARGS]):
    def __call__(
        self, batch: Iterator[Sequence[T]], length: int, *args: ARGS, **kwargs: KWARGS
    ) -> RT:
        """Process a series of batches

        :argument length: The number of batches
        """
        ...


class Processor(Protocol, Generic[T, ARGS, KWARGS]):
    def __call__(self, batch: Sequence[T], *args: ARGS, **kwargs: KWARGS) -> None:
        ...


class Reducer(Protocol, Generic[T, RT, ARGS, KWARGS]):
    def __call__(
        self, batch: Sequence[T], value: RT, *args: ARGS, **kwargs: KWARGS
    ) -> RT:
        ...


class BatcherWorker:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size #actually a micro-batch size, may be adapted dynamically

    def pre_train_setup(self, probe_fn: Callable, fabric) -> None:
        """Optional one-time pre-training setup (e.g. fitting a memory model).

        Trainers call this unconditionally after the worker is created and before
        the model is DDP-wrapped. The default is a no-op; predictive batchers
        override it to run their probe phase.
        """
        return

    def process_withreplay(
        self,
        batch: Sequence[T],
        process: IterativeProcessor[T, RT, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS,
    ) -> RT:
        """Process a batch with replay

        Replay = if an error occurs, the full batch is re-processed
        """
        return process(iter([batch]), 1, *args, **kwargs)

    def process(
        self,
        batch: Sequence[T],
        process: Processor[T, ARGS, KWARGS],
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> None:
        """Process a batch

        If a recoverable OOM error occurs, the processing continues but do not
        reprocess previously processed samples.

        Arguments:
            batch: The data to process
            process: The processing function
            raise_oom: Raise an OOM exception when an OOM is recoverable instead
            of continuing
        """
        process(batch, *args, **kwargs)

    def reduce(
        self,
        batch: Sequence[T],
        reducer: Reducer[T, RT, ARGS, KWARGS],
        initialvalue: RT,
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> RT:
        """Attributes:

        Arguments:
            batch: The data to process

            reducer: The reducer function, whose two first arguments are a slice
            of T and the reduced value, and that returns a new value

            raise_oom: Raise an OOM exception when an OOM is recoverable instead
            of continuing
        """
        return reducer(batch, initialvalue, *args, **kwargs)

    async def aio_reduce(
        self,
        batch: Sequence[T],
        reducer: Reducer[T, RT, ARGS, KWARGS],
        initialvalue: RT,
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> RT:
        """Attributes:

        Arguments:
            batch: The data to process

            reducer: The reducer function, whose two first arguments are a slice
            of T and the reduced value, and that returns a new value

            raise_oom: Raise an OOM exception when an OOM is recoverable instead
            of continuing
        """
        return await reducer(batch, initialvalue, *args, **kwargs)


def is_cublas_alloc_failed(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUBLAS_STATUS_ALLOC_FAILED" in exception.args[0]
    )


def is_oom_error(exception):
    """Detect CUDA out-of-memory errors"""
    from pytorch_lightning.utilities.memory import is_oom_error as legacy_check

    if legacy_check(exception):
        return True

    return is_cublas_alloc_failed(exception)


class Batcher(Config):
    """Responsible for micro-batching when the batch does not fit in memory

    The base class just does nothing (no adaptation)
    """

    def initialize(self, batch_size: int) -> BatcherWorker:
        logger.info("Using a simple batcher with batch size %d", batch_size)
        return BatcherWorker(batch_size)


class PowerAdaptativeBatcherWorker(BatcherWorker):
    def __init__(self, batch_size: int):
        super().__init__(batch_size)
        self.max_batch_size = batch_size
        self.current_divider = 1
        logger.info("[xpm_torch] Adaptative batcher: initial batch size is %d", self.batch_size)

    def get_ranges(self, batch_size):
        ranges = []
        ix = 0
        while ix < batch_size:
            ranges.append(slice(ix, ix + self.batch_size))
            ix += self.batch_size
        return ranges

    def iter(self, batch: Sequence[T], ranges: List[slice]) -> Iterator[Sequence[T]]:
        for range in ranges:
            yield batch[range]

    def process_withreplay(
        self,
        batch: Sequence[T],
        process: IterativeProcessor[T, RT, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS,
    ) -> RT:
        while True:
            ranges = self.get_ranges(len(batch))
            rt = self._run(
                lambda: process(self.iter(batch, ranges), len(ranges), *args, **kwargs)
            )
            if not isinstance(rt, RecoverableOOMError):
                return rt

    def process(
        self,
        batch: Sequence[T],
        process: Processor[T, ARGS, KWARGS],
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> None:
        ix = 0
        while ix < len(batch):
            s = slice(ix, ix + self.batch_size)
            rt = self._run(lambda: process(batch[s], *args, **kwargs))
            if not isinstance(rt, RecoverableOOMError):
                ix += self.batch_size
            elif raise_oom:
                raise rt

    def reduce(
        self,
        batch: Sequence[T],
        reducer: Reducer[T, RT, ARGS, KWARGS],
        value: RT,
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> RT:
        """
        Reduce a batch using the process function

        Args:
            batch: A batch
            reducer: A function that

        raise_oom: Raise an OOM exception when an OOM is recoverable
        """
        ix = 0
        while ix < len(batch):
            s = slice(ix, ix + self.batch_size)
            rt = self._run(lambda: reducer(batch[s], value, *args, **kwargs))
            if not isinstance(rt, RecoverableOOMError):
                ix += self.batch_size
                value = rt
            elif raise_oom:
                raise RecoverableOOMError()

        return value

    def _run(self, process: Callable[[], RT]) -> Union[RecoverableOOMError, RT]:
        try:
            # Perform a process
            return process()
        except RuntimeError as exception:
            if is_oom_error(exception):
                garbage_collection_cuda()
                while True:
                    self.current_divider += 1
                    new_batchsize = self.max_batch_size // self.current_divider
                    if new_batchsize != self.batch_size:
                        break

                self.batch_size = new_batchsize
                if self.batch_size == 0:
                    logger.error("Cannot decrease batch size below 1")
                    raise
                logger.info(
                    "Adaptative batcher: reducing batch size to %d", self.batch_size
                )
                return RecoverableOOMError(f"Reducing batch size to {self.batch_size}")
            else:
                # Other exception
                raise


class PowerAdaptativeBatcher(Batcher):
    """Starts with the provided batch size, and then divides in 2, 3, etc.
    until there is no more OOM
    """

    def initialize(self, batch_size: int) -> PowerAdaptativeBatcherWorker:
        return PowerAdaptativeBatcherWorker(batch_size)


# ---------------------------------------------------------------------------
# Predictive batching: fit a user-supplied memory-cost formula from probes
# run before DDP wrapping, then use it to choose the largest safe microbatch
# at train time. Avoids OOM-reactive behavior, which deadlocks under DDP.
# ---------------------------------------------------------------------------


class ProbePoint(Config):
    """One probe configuration: a mapping {variable_name -> value} for a single
    point at which to measure peak GPU memory during the probe phase.
    """

    values: Param[dict[str, int]]


class BatchDimsProvider(Config):
    """Contract between a `PredictiveBatcher` and its surrounding trainer.

    Implementations are responsible for:
      - reading the formula's input variables from a runtime batch
      - constructing a representative probe batch matching given dims

    `probe_dims` lives here (and not on the batcher) because probes are
    intrinsically tied to the data shape the provider knows how to construct.
    """

    probe_dims: Param[set[ProbePoint]]

    @abstractmethod
    def compute_batch_dims(self, batch) -> dict[str, int]:
        """Return the formula variables (including the batch-size variable) for
        this runtime batch."""
        ...

    @abstractmethod
    def build_probe_batch(self, dims: dict[str, int]):
        """Construct a batch (synthetic or sampled) matching the given probe
        dims. Used during the pre-training probe phase only."""
        ...


def _canonicalize_formula(
    formula: str,
    coefficient_names: set[str],
    variable_names: set[str],
) -> str:
    """Parse and rewrite a memory-cost formula to a canonical string.

    Ensures the experimaestro task identifier is stable across equivalent
    inputs (whitespace, term order, etc.). Also validates that the formula is
    linear in the declared coefficients.
    """
    import sympy

    coef_syms = {n: sympy.Symbol(n) for n in coefficient_names}
    var_syms = {n: sympy.Symbol(n) for n in variable_names}
    locals_ = {**coef_syms, **var_syms}

    try:
        expr = sympy.sympify(formula, locals=locals_)
    except Exception as e:
        raise ValueError(f"Failed to parse formula {formula!r}: {e}") from e

    expr = sympy.expand(expr)

    declared = set(coef_syms.values()) | set(var_syms.values())
    extra = expr.free_symbols - declared
    if extra:
        raise ValueError(
            f"Formula {formula!r} uses undeclared symbols: "
            f"{sorted(str(s) for s in extra)}. Declared coefficients="
            f"{sorted(coefficient_names)}, variables={sorted(variable_names)}."
        )

    # Group expanded terms by which coefficient (if any) each contains, while
    # checking linearity. groups[None] holds the constant (no-coefficient) term.
    groups: dict[Optional[str], sympy.Expr] = {}
    for term in sympy.Add.make_args(expr):
        present = [n for n, s in coef_syms.items() if s in term.free_symbols]
        if len(present) > 1:
            raise ValueError(
                f"Formula {formula!r}: term {term} contains multiple coefficients "
                f"{present}; formulas must be linear in coefficients."
            )
        if len(present) == 1:
            cn = present[0]
            cs = coef_syms[cn]
            poly = sympy.Poly(term, cs)
            # term must be coef^1 * monomial
            if poly.degree() != 1:
                raise ValueError(
                    f"Formula {formula!r}: term {term} is not linear in {cn} "
                    f"(degree {poly.degree()})."
                )
            monomial = poly.all_coeffs()[0]
            if any(s in monomial.free_symbols for s in coef_syms.values()):
                raise ValueError(
                    f"Formula {formula!r}: term {term} mixes coefficient symbols."
                )
        else:
            cn = None
            monomial = term

        groups[cn] = groups.get(cn, sympy.S.Zero) + monomial

    # Render each grouped monomial as a canonical polynomial in the variables,
    # sorted descending lexicographically by exponent tuple.
    var_sorted = sorted(variable_names)
    var_syms_sorted = [sympy.Symbol(v) for v in var_sorted]

    def render_monomial(mono: "sympy.Expr") -> str:
        mono = sympy.expand(mono)
        if mono == 0:
            return "0"
        if not var_syms_sorted:
            return str(mono)
        poly = sympy.Poly(mono, *var_syms_sorted)
        terms = sorted(poly.terms(), key=lambda t: t[0], reverse=True)
        parts = []
        for exps, coeff in terms:
            factors: List[str] = []
            if coeff != 1 or all(e == 0 for e in exps):
                factors.append(str(coeff))
            for var, e in zip(var_sorted, exps):
                if e == 1:
                    factors.append(var)
                elif e > 1:
                    factors.append(f"{var}**{e}")
            if not factors:
                factors = ["1"]
            parts.append("*".join(factors))
        return " + ".join(parts)

    parts: List[str] = []
    for cn in sorted(c for c in groups if c is not None):
        mono = sympy.expand(groups[cn])
        if mono == 0:
            continue
        if mono == 1:
            parts.append(cn)
        else:
            mono_str = render_monomial(mono)
            # Wrap multi-term monomials in parens; single-term stays unwrapped.
            wrap = " + " in mono_str or " - " in mono_str
            parts.append(f"{cn}*({mono_str})" if wrap else f"{cn}*{mono_str}")

    if None in groups:
        const = sympy.expand(groups[None])
        if const != 0:
            parts.append(render_monomial(const))

    return " + ".join(parts) if parts else "0"


class MicroBatchPacker(Config):
    """Strategy for splitting an outer batch into microbatches under a
    `PredictiveBatcherWorker`.

    Different implementations trade off simplicity against packing density.
    Plug into a `PredictiveBatcher` via its `packer` Param.
    """

    @abstractmethod
    def pack(
        self,
        records: Sequence[T],
        worker: "PredictiveBatcherWorker",
    ) -> Iterator[Sequence[T]]:
        """Yield microbatches from `records`. Each yielded microbatch must
        respect the worker's current memory budget."""
        ...


class UniformPacker(MicroBatchPacker):
    """Default packer: compute dims once from the outer batch, choose a single
    microbatch size, slice equally.

    Conservative: each microbatch is sized as if it contained the
    worst-case-cost record from the outer batch. For homogeneous batches
    (all records same shape) this is optimal; for heterogeneous batches
    `AdaptivePacker` packs denser.
    """

    def pack(
        self,
        records: Sequence[T],
        worker: "PredictiveBatcherWorker",
    ) -> Iterator[Sequence[T]]:
        n = len(records)
        if n == 0:
            return
        dims = worker.batcher.dims_provider.compute_batch_dims(records)
        bsvar = worker.batcher.batch_size_variable
        upper = worker.max_batch_size
        if bsvar in dims:
            upper = min(upper, int(dims[bsvar]))
        upper = min(upper, n)
        budget = worker.current_budget_bytes()
        mbs = max(worker.choose_batch_size(dims, budget, upper_bound=upper), 1)
        ix = 0
        while ix < n:
            yield records[ix : ix + mbs]
            ix += mbs


class AdaptivePacker(MicroBatchPacker):
    """Pack records greedily: incrementally accumulate into a microbatch and
    emit when adding one more record would exceed the memory budget.

    Two improvements over `UniformPacker`:

    * Each microbatch is sized for its own dims (recomputed per-microbatch),
      so chunks containing only short-cost records can be much larger.
    * Optionally sort by a per-record cost proxy (`sort_by`) descending so
      heavy records go into one early microbatch and light records pack
      densely into later ones.
    """

    sort_by: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Variable name (must be produced by `compute_batch_dims([record])`) to
    sort records by, descending. Reduces microbatch count for batches with
    heterogeneous per-record cost. Default: no sort (preserve sampler order)."""

    def pack(
        self,
        records: Sequence[T],
        worker: "PredictiveBatcherWorker",
    ) -> Iterator[Sequence[T]]:
        if len(records) == 0:
            return

        provider = worker.batcher.dims_provider
        bsvar = worker.batcher.batch_size_variable
        budget = worker.current_budget_bytes()
        upper = worker.max_batch_size

        items: List = list(records)
        if self.sort_by is not None:
            sort_key = self.sort_by
            items.sort(
                key=lambda r: int(provider.compute_batch_dims([r]).get(sort_key, 0)),
                reverse=True,
            )

        current: List = []

        def fits(candidate: List) -> bool:
            if len(candidate) > upper:
                return False
            dims = provider.compute_batch_dims(candidate)
            # Ensure bs in dims matches actual candidate length for prediction.
            dims = {**dims, bsvar: len(candidate)}
            return worker.predict(dims) <= budget

        for record in items:
            candidate = current + [record]
            if fits(candidate):
                current = candidate
                continue
            if current:
                yield current
                current = []
            # Test the lone record. If even one record exceeds budget, emit it
            # alone with a warning — there's nothing smaller we can do.
            if fits([record]):
                current = [record]
            else:
                dims = provider.compute_batch_dims([record])
                logger.warning(
                    "AdaptivePacker: single record exceeds budget "
                    "(dims=%s, predicted=%.2e, budget=%.2e). Emitting anyway.",
                    {**dims, bsvar: 1},
                    worker.predict({**dims, bsvar: 1}),
                    budget,
                )
                yield [record]
                current = []
        if current:
            yield current


class PredictiveBatcher(Batcher):
    """Predictive batcher.

    Fits a user-supplied memory cost formula linear in coefficients (e.g.
    `"a*bs*l**2 + b*bs*l + c"`) from a small set of probes run **before DDP
    wrapping**, then chooses the largest microbatch that fits the remaining
    GPU memory budget at train time. Avoids the multi-GPU NCCL hang that an
    OOM-reactive batcher causes when one rank shrinks its microbatch and
    desynchronizes from the others.

    Important: the trainer must invoke `pre_train_setup` on the worker before
    the model is wrapped with `fabric.setup(model)`. `LossTrainer` does this in
    its `initialize()`.
    """

    formula: Param[str]
    """Cost formula, linear in coefficients. Variables and coefficients must be
    declared via `variable_names` and `coefficient_names`. Canonicalized in
    `__validate__` so equivalent inputs share an identifier."""

    coefficient_names: Param[set[str]]
    """Names of the unknown coefficients in `formula`, fit via least squares."""

    variable_names: Param[set[str]]
    """Names of the input variables in `formula` (e.g. `{"bs", "l"}`)."""

    batch_size_variable: Param[str] = field(default="bs")
    """Which variable in `variable_names` is the microbatch size to maximize."""

    memory_fraction: Param[float] = field(default=0.75)
    """Safety margin on free GPU memory. Default is conservative to absorb DDP
    gradient buckets / NCCL buffers that the pre-DDP probe doesn't measure."""

    extra_overhead_bytes: Param[int] = field(default=0, ignore_default=True)
    """Additional bytes to subtract from the budget at train time, in case
    `memory_fraction` alone isn't enough on a given setup."""

    min_batch_size: Param[int] = field(default=1)
    """Floor on the chosen microbatch size."""

    dims_provider: Param[BatchDimsProvider]
    """How to read dims from a runtime batch and how to build a probe batch."""

    packer: Param[MicroBatchPacker] = field(
        default_factory=lambda: UniformPacker.C(),
        ignore_default=True,
    )
    """How to split an outer batch into microbatches. Default `UniformPacker`
    matches conservative slicing on outer-batch dims; use `AdaptivePacker` for
    heterogeneous-cost batches (e.g. variable sequence length)."""

    def __validate__(self):
        self.formula = _canonicalize_formula(
            self.formula, self.coefficient_names, self.variable_names
        )
        if self.batch_size_variable not in self.variable_names:
            raise ValueError(
                f"batch_size_variable={self.batch_size_variable!r} not in "
                f"variable_names={sorted(self.variable_names)}"
            )
        if len(self.dims_provider.probe_dims) == 0:
            raise ValueError(
                "PredictiveBatcher.dims_provider.probe_dims is empty; need at "
                "least one probe point to fit coefficients."
            )
        for probe in self.dims_provider.probe_dims:
            probe_vars = set(probe.values.keys())
            missing = self.variable_names - probe_vars
            extra = probe_vars - self.variable_names
            if missing or extra:
                raise ValueError(
                    f"Probe dims {probe.values} don't match formula variables "
                    f"{sorted(self.variable_names)}: missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
        if not (0 < self.memory_fraction <= 1):
            raise ValueError(
                f"memory_fraction must be in (0, 1], got {self.memory_fraction}"
            )

    def initialize(self, batch_size: int) -> "PredictiveBatcherWorker":
        return PredictiveBatcherWorker(batch_size, self)


class PredictiveBatcherWorker(BatcherWorker):
    """Worker for `PredictiveBatcher`.

    Holds the fitted coefficients and overrides `process` / `reduce` /
    `process_withreplay` so the formula-driven batching IS the batching: there
    is no silent fixed-size fallback through the same API.

    A real OOM at train time is propagated (not silently retried) — silent
    retry would put us back in OOM-reactive territory and reintroduce the
    multi-GPU NCCL-hang risk this design exists to avoid.
    """

    def __init__(self, batch_size: int, batcher: "PredictiveBatcher"):
        super().__init__(batch_size)
        self.batcher = batcher
        self.max_batch_size = batch_size
        self._observations: List[tuple[dict[str, int], float]] = []
        self._coefficients: Optional[np.ndarray] = None
        self._term_specs: Optional[List[tuple[Optional[str], dict[str, int]]]] = None
        self._coef_order: Optional[List[str]] = None
        logger.info(
            "[xpm_torch] Predictive batcher: initial (max) batch size is %d",
            batch_size,
        )

    # ----- formula parsing / evaluation -------------------------------------

    def _parse_terms(self) -> None:
        """Decompose the canonical formula into (coef_name_or_None, monomial_exponents).

        Called lazily so we don't pay the sympy cost in workers that never get
        used (e.g. construction-only tests).
        """
        import sympy

        coef_syms = {n: sympy.Symbol(n) for n in self.batcher.coefficient_names}
        var_syms = {n: sympy.Symbol(n) for n in self.batcher.variable_names}
        locals_ = {**coef_syms, **var_syms}
        expr = sympy.expand(sympy.sympify(self.batcher.formula, locals=locals_))

        var_sorted = sorted(self.batcher.variable_names)
        var_syms_sorted = [sympy.Symbol(v) for v in var_sorted]

        terms: List[tuple[Optional[str], dict[str, int]]] = []
        for term in sympy.Add.make_args(expr):
            present = [n for n, s in coef_syms.items() if s in term.free_symbols]
            if len(present) > 1:
                raise ValueError(f"Non-linear term {term}")
            if len(present) == 1:
                cn = present[0]
                cs = coef_syms[cn]
                monomial = sympy.Poly(term, cs).all_coeffs()[0]
            else:
                cn = None
                monomial = term
            poly = sympy.Poly(sympy.expand(monomial), *var_syms_sorted)
            poly_terms = poly.terms()
            # Each grouped monomial is itself a sum of var-monomials; record each.
            for exps, coeff in poly_terms:
                exp_map = {v: int(e) for v, e in zip(var_sorted, exps) if int(e) != 0}
                # bundle the scalar coefficient into the term spec: store as a
                # tuple (coef_name, exp_map, scalar). For canonical formulas
                # scalar will typically be 1, but coefficients may have been
                # combined (e.g. a*bs + a*bs would canonicalize to a*(2*bs))
                terms.append((cn, exp_map, float(coeff)))

        self._term_specs = terms
        self._coef_order = sorted(self.batcher.coefficient_names)

    def _design_row(self, dims: dict[str, int]) -> np.ndarray:
        """Build one row of the design matrix for the given dims.

        Columns are in `_coef_order`, with a trailing constant column.
        """
        assert self._term_specs is not None and self._coef_order is not None
        ncoefs = len(self._coef_order)
        row = np.zeros(ncoefs + 1, dtype=np.float64)
        for coef_name, exp_map, scalar in self._term_specs:
            value = scalar
            for var, e in exp_map.items():
                value *= float(dims[var]) ** e
            if coef_name is None:
                row[ncoefs] += value
            else:
                idx = self._coef_order.index(coef_name)
                row[idx] += value
        return row

    # ----- public API -------------------------------------------------------

    def observe(self, dims: dict[str, int], peak_memory_bytes: float) -> None:
        if self._term_specs is None:
            self._parse_terms()
        self._observations.append((dict(dims), float(peak_memory_bytes)))

    def fit(self) -> None:
        if self._term_specs is None:
            self._parse_terms()
        if len(self._observations) == 0:
            raise RuntimeError(
                "PredictiveBatcherWorker.fit() called with no observations"
            )
        X = np.stack([self._design_row(d) for d, _ in self._observations])
        y = np.array([m for _, m in self._observations], dtype=np.float64)
        # lstsq returns (solution, residuals, rank, singular_values)
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        # Last entry is the constant term coefficient (not user-named); we keep
        # it implicit in the design and store the full vector here.
        self._coefficients = sol
        logger.info(
            "[xpm_torch] Predictive batcher fitted: coefficients=%s (order=%s + const)",
            sol.tolist(),
            self._coef_order,
        )

    def is_fitted(self) -> bool:
        return self._coefficients is not None

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        return self._coefficients

    def set_coefficients(self, coeffs: np.ndarray) -> None:
        if self._term_specs is None:
            self._parse_terms()
        self._coefficients = np.asarray(coeffs, dtype=np.float64)

    def predict(self, dims: dict[str, int]) -> float:
        if not self.is_fitted():
            raise RuntimeError("PredictiveBatcherWorker.predict before fit/broadcast")
        return float(self._design_row(dims) @ self._coefficients)

    def choose_batch_size(
        self,
        dims: dict[str, int],
        budget_bytes: float,
        *,
        upper_bound: Optional[int] = None,
    ) -> int:
        """Largest bs in `[min_batch_size, upper_bound]` whose predicted memory
        is <= `budget_bytes`. `upper_bound` defaults to `self.max_batch_size`."""
        if not self.is_fitted():
            raise RuntimeError(
                "PredictiveBatcherWorker.choose_batch_size before fit/broadcast"
            )
        bsvar = self.batcher.batch_size_variable
        lo = self.batcher.min_batch_size
        hi = self.max_batch_size if upper_bound is None else upper_bound
        if lo > hi:
            return lo

        def fits(bs: int) -> bool:
            d = dict(dims)
            d[bsvar] = bs
            return self.predict(d) <= budget_bytes

        if not fits(lo):
            logger.warning(
                "Predictive batcher: even min_batch_size=%d exceeds budget=%.2e "
                "(dims=%s, predicted=%.2e). Using min_batch_size anyway.",
                lo, budget_bytes, dims, self.predict({**dims, bsvar: lo}),
            )
            return lo
        if fits(hi):
            return hi
        # Bisection on integer interval [lo, hi]; invariant: fits(lo), not fits(hi).
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def current_budget_bytes(self) -> float:
        """Current GPU memory budget in bytes (free * memory_fraction - overhead).
        Public so `MicroBatchPacker` implementations can read it."""
        if not torch.cuda.is_available():
            # No GPU memory limit applies; use a huge budget. Predictive batching
            # without CUDA is degenerate but shouldn't crash.
            return float("inf")
        device = torch.cuda.current_device()
        free, _total = torch.cuda.mem_get_info(device)
        budget = free * self.batcher.memory_fraction - self.batcher.extra_overhead_bytes
        return max(budget, 0.0)

    def _pack(self, records):
        """Yield microbatches by delegating to the configured packer."""
        if not self.is_fitted():
            raise RuntimeError(
                "PredictiveBatcherWorker called before fit/broadcast; trainer "
                "must call pre_train_setup() first."
            )
        return self.batcher.packer.pack(records, self)

    # ----- pre-train setup --------------------------------------------------

    def pre_train_setup(self, probe_fn: Callable, fabric) -> None:
        """Run the probe phase on rank 0, fit, then broadcast coefficients.

        Must be called BEFORE `fabric.setup(model)` — running probes on rank 0
        alone after the model is DDP-wrapped would deadlock on the first
        gradient-bucket all_reduce.
        """
        provider = self.batcher.dims_provider
        device = fabric.device if fabric is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        is_rank_zero = (fabric is None) or (fabric.global_rank == 0)
        world_size = (fabric.world_size if fabric is not None else 1)

        if is_rank_zero:
            logger.info(
                "[xpm_torch] Predictive batcher: probing %d points on rank 0",
                len(provider.probe_dims),
            )
            for probe in provider.probe_dims:
                dims = dict(probe.values)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    probe_batch = provider.build_probe_batch(dims)
                    probe_fn(probe_batch)
                except RuntimeError as e:
                    if is_oom_error(e):
                        logger.warning(
                            "Predictive batcher: probe OOM at dims=%s, skipping. (%s)",
                            dims, e,
                        )
                        garbage_collection_cuda()
                        continue
                    raise
                peak = (
                    float(torch.cuda.max_memory_allocated(device))
                    if torch.cuda.is_available()
                    else 0.0
                )
                logger.info(
                    "[xpm_torch] Predictive probe: dims=%s peak=%.2e bytes",
                    dims, peak,
                )
                self.observe(dims, peak)
            self.fit()

        # Broadcast coefficients (and constant column) from rank 0 to all ranks.
        if fabric is not None and world_size > 1:
            if is_rank_zero:
                assert self._coefficients is not None
                tensor = torch.as_tensor(self._coefficients, dtype=torch.float64)
            else:
                # Other ranks must know the tensor length; parse formula locally
                # to determine ncoefs + 1.
                if self._term_specs is None:
                    self._parse_terms()
                ncoefs = len(self._coef_order) + 1
                tensor = torch.zeros(ncoefs, dtype=torch.float64)
            tensor = tensor.to(device)
            tensor = fabric.broadcast(tensor, src=0)
            if not is_rank_zero:
                self.set_coefficients(tensor.detach().cpu().numpy())
            fabric.barrier()
        elif fabric is not None:
            fabric.barrier()

    # ----- overridden microbatching methods ---------------------------------

    def process(
        self,
        batch: Sequence[T],
        process: "Processor[T, ARGS, KWARGS]",
        *args,
        raise_oom: bool = False,
        **kwargs,
    ) -> None:
        for microbatch in self._pack(batch):
            process(microbatch, *args, **kwargs)

    def reduce(
        self,
        batch: Sequence[T],
        reducer: "Reducer[T, RT, ARGS, KWARGS]",
        value: RT,
        *args,
        raise_oom: bool = False,
        **kwargs,
    ) -> RT:
        for microbatch in self._pack(batch):
            value = reducer(microbatch, value, *args, **kwargs)
        return value

    def process_withreplay(
        self,
        batch: Sequence[T],
        process: "IterativeProcessor[T, RT, ARGS, KWARGS]",
        *args,
        **kwargs,
    ) -> RT:
        microbatches = list(self._pack(batch))
        return process(iter(microbatches), len(microbatches), *args, **kwargs)

    async def aio_reduce(
        self,
        batch: Sequence[T],
        reducer: "Reducer[T, RT, ARGS, KWARGS]",
        initialvalue: RT,
        *args,
        raise_oom: bool = False,
        **kwargs,
    ) -> RT:
        value = initialvalue
        for microbatch in self._pack(batch):
            value = await reducer(microbatch, value, *args, **kwargs)
        return value
