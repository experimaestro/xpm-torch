"""Tests for the predictive batcher: formula canonicalization, fitting, and
microbatch selection. Pure-CPU / no-CUDA tests — the GPU-bound code paths are
exercised through dependency injection.
"""

import numpy as np
import pytest
from experimaestro import sealed_set

from xpm_torch.batchers import (
    AdaptivePacker,
    BatchDimsProvider,
    PredictiveBatcher,
    ProbePoint,
    UniformPacker,
    _canonicalize_formula,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class DummyDimsProvider(BatchDimsProvider):
    """Records dims via len(batch) and a fixed l in attributes for tests."""

    def compute_batch_dims(self, batch) -> dict[str, int]:
        return {"bs": len(batch), "l": batch[0] if batch else 0}

    def build_probe_batch(self, dims: dict[str, int]):
        # represent a "batch" as a list whose length is bs; each element is l
        return [dims["l"]] * dims["bs"]


def make_batcher(formula="a*bs*l**2 + b*bs*l + c", probes=None):
    if probes is None:
        probes = sealed_set(
            ProbePoint.C(values={"bs": 1, "l": 64}),
            ProbePoint.C(values={"bs": 1, "l": 128}),
            ProbePoint.C(values={"bs": 2, "l": 128}),
            ProbePoint.C(values={"bs": 4, "l": 64}),
        )
    return PredictiveBatcher.C(
        formula=formula,
        coefficient_names={"a", "b", "c"},
        variable_names={"bs", "l"},
        batch_size_variable="bs",
        dims_provider=DummyDimsProvider.C(probe_dims=probes),
    )


# ---------------------------------------------------------------------------
# canonicalization
# ---------------------------------------------------------------------------


def test_canonicalize_equivalent_inputs_produce_same_string():
    variants = [
        "a*bs*l**2 + b*bs*l + c",
        "c + a*bs*l**2 + b*bs*l",
        "a *bs* l**2+b*bs*l+ c",
        "c + b*bs*l + a*l**2*bs",
        "a*l**2*bs + c + b*l*bs",
    ]
    results = {
        _canonicalize_formula(v, {"a", "b", "c"}, {"bs", "l"}) for v in variants
    }
    assert len(results) == 1, f"non-canonical outputs: {results}"


def test_canonicalize_rejects_nonlinear_in_coefficients():
    with pytest.raises(ValueError, match="multiple coefficients|linear"):
        _canonicalize_formula("a*b*bs", {"a", "b"}, {"bs"})


def test_canonicalize_rejects_undeclared_symbols():
    with pytest.raises(ValueError, match="undeclared"):
        _canonicalize_formula("a*bs + d", {"a"}, {"bs"})


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_empty_probes_fails():
    with pytest.raises(ValueError, match="probe_dims is empty"):
        PredictiveBatcher.__validate__(make_batcher(probes=sealed_set()))


def test_validate_variable_mismatch_fails():
    bad_probes = sealed_set(ProbePoint.C(values={"l": 64}))
    with pytest.raises(ValueError, match="don't match formula variables"):
        PredictiveBatcher.__validate__(make_batcher(probes=bad_probes))


def test_validate_happy_path():
    # Should not raise.
    PredictiveBatcher.__validate__(make_batcher())


def test_validate_canonicalizes_formula_in_place():
    b = make_batcher(formula="c + b*l*bs + a*l**2*bs")
    PredictiveBatcher.__validate__(b)
    assert b.formula == "a*bs*l**2 + b*bs*l + c"


# ---------------------------------------------------------------------------
# fit + predict + choose_batch_size
# ---------------------------------------------------------------------------


def _fit_worker_from_truth(coeffs_true, dims_points):
    """Build a worker whose observations follow `y = formula(true_coefs, dims)`
    exactly, then call fit() so it can be queried."""
    batcher = make_batcher()
    PredictiveBatcher.__validate__(batcher)  # canonicalize
    worker = batcher.initialize(batch_size=64)

    # Synthetic observations: a=ca, b=cb, c=cc; y = ca*bs*l^2 + cb*bs*l + cc
    ca, cb, cc = coeffs_true
    for d in dims_points:
        y = ca * d["bs"] * d["l"] ** 2 + cb * d["bs"] * d["l"] + cc
        worker.observe(d, y)
    worker.fit()
    return worker


def test_fit_recovers_known_coefficients():
    dims_points = [
        {"bs": 1, "l": 64},
        {"bs": 1, "l": 128},
        {"bs": 2, "l": 128},
        {"bs": 4, "l": 64},
        {"bs": 2, "l": 256},
    ]
    truth = (2.0, 3.0, 100.0)
    worker = _fit_worker_from_truth(truth, dims_points)
    # Order in coefficients: sorted(coefficient_names) + const
    # → [a, b, c] + const; but `c` is itself a coefficient (intercept-like) here.
    # The design uses `_coef_order = sorted({a,b,c}) = [a,b,c]` and trailing
    # constant column captures terms with no coefficient (there are none here).
    assert worker._coef_order == ["a", "b", "c"]
    np.testing.assert_allclose(worker.coefficients[:3], truth, atol=1e-6)
    # Constant column should be ~0 since the formula has no no-coefficient term.
    assert abs(worker.coefficients[3]) < 1e-6


def test_predict_matches_formula():
    truth = (2.0, 3.0, 100.0)
    dims_points = [
        {"bs": 1, "l": 64},
        {"bs": 1, "l": 128},
        {"bs": 2, "l": 128},
        {"bs": 4, "l": 64},
    ]
    worker = _fit_worker_from_truth(truth, dims_points)
    d = {"bs": 8, "l": 256}
    expected = 2.0 * 8 * 256**2 + 3.0 * 8 * 256 + 100.0
    assert abs(worker.predict(d) - expected) < 1e-3


def test_choose_batch_size_respects_budget():
    truth = (2.0, 3.0, 100.0)
    dims_points = [
        {"bs": 1, "l": 64},
        {"bs": 1, "l": 128},
        {"bs": 2, "l": 128},
        {"bs": 4, "l": 64},
    ]
    worker = _fit_worker_from_truth(truth, dims_points)

    # Pick a budget that strictly fits bs=10 but not bs=11.
    dims = {"bs": 0, "l": 128}  # bs irrelevant — choose_batch_size overrides it
    bs10_mem = 2.0 * 10 * 128**2 + 3.0 * 10 * 128 + 100.0
    bs11_mem = 2.0 * 11 * 128**2 + 3.0 * 11 * 128 + 100.0
    budget = (bs10_mem + bs11_mem) / 2  # midway: fits 10, not 11
    chosen = worker.choose_batch_size(dims, budget, upper_bound=64)
    assert chosen == 10, f"expected 10, got {chosen}"

    # Below-min budget: returns min_batch_size with a warning.
    chosen = worker.choose_batch_size(dims, 0.0, upper_bound=64)
    assert chosen == worker.batcher.min_batch_size


def test_set_coefficients_round_trip():
    truth = (2.0, 3.0, 100.0)
    dims_points = [{"bs": 1, "l": 64}, {"bs": 2, "l": 128}, {"bs": 4, "l": 64}]
    worker = _fit_worker_from_truth(truth, dims_points)
    coeffs = worker.coefficients.copy()

    # Make a fresh, unfitted worker and inject the coefficients.
    batcher = make_batcher()
    PredictiveBatcher.__validate__(batcher)
    fresh = batcher.initialize(batch_size=64)
    assert not fresh.is_fitted()
    fresh.set_coefficients(coeffs)
    assert fresh.is_fitted()
    np.testing.assert_allclose(fresh.coefficients, coeffs)


def test_overridden_process_requires_fit():
    batcher = make_batcher()
    PredictiveBatcher.__validate__(batcher)
    worker = batcher.initialize(batch_size=8)
    with pytest.raises(RuntimeError, match="before fit/broadcast"):
        worker.process([64, 64, 64, 64], lambda b: None)


# ---------------------------------------------------------------------------
# Packers
# ---------------------------------------------------------------------------


class HeterogeneousDimsProvider(BatchDimsProvider):
    """Each record is an int representing its `l` (per-record cost proxy).
    The batch's `l` dim is `max(records)` — matches transformer padding."""

    def compute_batch_dims(self, batch) -> dict[str, int]:
        return {"bs": len(batch), "l": max(batch) if batch else 0}

    def build_probe_batch(self, dims: dict[str, int]):
        return [dims["l"]] * dims["bs"]


def _make_hetero_batcher(packer=None):
    probes = sealed_set(
        ProbePoint.C(values={"bs": 1, "l": 64}),
        ProbePoint.C(values={"bs": 1, "l": 128}),
        ProbePoint.C(values={"bs": 2, "l": 128}),
        ProbePoint.C(values={"bs": 4, "l": 64}),
    )
    kwargs = dict(
        formula="a*bs*l**2 + b*bs*l + c",
        coefficient_names={"a", "b", "c"},
        variable_names={"bs", "l"},
        batch_size_variable="bs",
        dims_provider=HeterogeneousDimsProvider.C(probe_dims=probes),
    )
    if packer is not None:
        kwargs["packer"] = packer
    return PredictiveBatcher.C(**kwargs)


def _fit_hetero_worker(batcher, truth=(2.0, 3.0, 100.0), batch_size=64,
                       budget=None, max_batch_size=None):
    """Build and manually fit a worker against a synthetic ground truth, then
    monkeypatch its budget so packing decisions are deterministic."""
    PredictiveBatcher.__validate__(batcher)
    worker = batcher.initialize(batch_size=batch_size)
    ca, cb, cc = truth
    points = [{"bs": 1, "l": 64}, {"bs": 2, "l": 128}, {"bs": 4, "l": 256}]
    for d in points:
        y = ca * d["bs"] * d["l"] ** 2 + cb * d["bs"] * d["l"] + cc
        worker.observe(d, y)
    worker.fit()
    if max_batch_size is not None:
        worker.max_batch_size = max_batch_size
    if budget is not None:
        worker.current_budget_bytes = lambda b=budget: b
    return worker


def test_uniform_packer_uniform_slices():
    """UniformPacker uses outer-batch dims for every microbatch — conservative."""
    batcher = _make_hetero_batcher(packer=UniformPacker.C())
    # All records l=128. Budget allows bs=4 at l=128.
    truth = (1.0, 0.0, 0.0)  # cost = bs * l**2
    budget = 4 * 128**2 + 1.0  # fits 4 records of l=128
    worker = _fit_hetero_worker(batcher, truth=truth, budget=budget,
                                max_batch_size=16)
    records = [128] * 12
    microbatches = list(worker._pack(records))
    assert all(len(m) == 4 for m in microbatches[:-1]), microbatches
    assert sum(len(m) for m in microbatches) == 12


def test_adaptive_packer_fills_each_microbatch():
    """AdaptivePacker recomputes dims per microbatch — small-l chunks pack denser."""
    batcher = _make_hetero_batcher(packer=AdaptivePacker.C())
    # cost = bs * l**2; budget = 4 * 128**2 → at l=128, 4 fit; at l=32, 64 fit.
    truth = (1.0, 0.0, 0.0)
    budget = 4 * 128**2 + 1.0
    worker = _fit_hetero_worker(batcher, truth=truth, budget=budget,
                                max_batch_size=64)

    # Order: 4 records of l=128, then 16 records of l=32. Without sort, packing
    # in input order: first microbatch gets 4 l=128 records (filled), then the
    # next packs as many l=32 records as fit.
    records = [128] * 4 + [32] * 16
    microbatches = list(worker._pack(records))
    # First chunk: 4 records of l=128
    assert microbatches[0] == [128] * 4, microbatches
    # Remaining l=32 records pack into at most ceil(16 / N_per_chunk) chunks.
    rest = [r for m in microbatches[1:] for r in m]
    assert rest == [32] * 16
    # AdaptivePacker should fit MANY l=32 records per chunk: at l=32, budget
    # holds budget / 32**2 = 64 records → all 16 fit in one chunk.
    assert len(microbatches[1]) == 16, microbatches


def test_adaptive_packer_sort_by_reduces_microbatch_count():
    """Sorting by l descending lets short-l records pack densely after the long ones."""
    batcher_unsorted = _make_hetero_batcher(packer=AdaptivePacker.C())
    batcher_sorted = _make_hetero_batcher(packer=AdaptivePacker.C(sort_by="l"))
    truth = (1.0, 0.0, 0.0)
    budget = 4 * 128**2 + 1.0

    # Interleaved heterogeneous records — input order is bad for packing.
    records = [32, 128, 32, 128, 32, 32, 128, 32, 128, 32, 32, 32]

    w_unsorted = _fit_hetero_worker(batcher_unsorted, truth=truth,
                                    budget=budget, max_batch_size=64)
    w_sorted = _fit_hetero_worker(batcher_sorted, truth=truth,
                                  budget=budget, max_batch_size=64)

    unsorted_chunks = list(w_unsorted._pack(records))
    sorted_chunks = list(w_sorted._pack(records))

    # Sorted should produce fewer (or equal) microbatches.
    assert len(sorted_chunks) <= len(unsorted_chunks), (
        f"sorted={len(sorted_chunks)} not <= unsorted={len(unsorted_chunks)}"
    )
    # And cover the same multiset of records.
    flat_sorted = sorted(r for m in sorted_chunks for r in m)
    flat_unsorted = sorted(r for m in unsorted_chunks for r in m)
    assert flat_sorted == flat_unsorted == sorted(records)


def test_adaptive_packer_single_record_over_budget_emits_alone():
    """If a single record exceeds the budget, AdaptivePacker emits it alone
    (with a warning) — never silently drops data."""
    batcher = _make_hetero_batcher(packer=AdaptivePacker.C())
    truth = (1.0, 0.0, 0.0)
    # Budget too small even for one l=256 record (cost = 1 * 256**2 = 65536).
    budget = 1000.0
    worker = _fit_hetero_worker(batcher, truth=truth, budget=budget,
                                max_batch_size=64)
    records = [256, 256]
    microbatches = list(worker._pack(records))
    # Two singleton microbatches (each emitted as fallback).
    assert microbatches == [[256], [256]], microbatches


def test_packer_default_is_uniform():
    """Default packer is UniformPacker — preserves prior behavior for users who
    haven't set `packer` explicitly."""
    batcher = _make_hetero_batcher()  # no packer arg
    assert isinstance(batcher.packer, UniformPacker)
