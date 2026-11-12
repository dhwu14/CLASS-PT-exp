#!/usr/bin/env python3
# DW: validate exact-time matter, tracer, and multipole products and compare convergence runs.
"""Validate QCDM/QSCF exact-time EFTofLSS text outputs.

Examples
--------
python3 tools/validate_qcdm_pt.py output/run_qcdm_pt_pk_nl.dat
python3 tools/validate_qcdm_pt.py output/run_qcdm_pt_tracer_real.dat
python3 tools/validate_qcdm_pt.py output/run_qcdm_pt_multipoles.dat
python3 tools/validate_qcdm_pt.py --compare run_a.dat run_b.dat --column 2 \
    --kmin 0.01 --kmax 0.2
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Iterable, List


def load_table(path: pathlib.Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = [float(token) for token in stripped.split()]
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: non-numeric row") from exc
            if not row:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    width = len(rows[0])
    for index, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(
                f"{path}: inconsistent column count at numeric row {index}: "
                f"expected {width}, found {len(row)}"
            )
    return rows


def max_relative(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    result = 0.0
    for a, b in zip(values_a, values_b):
        scale = max(abs(a), abs(b), 1.0e-300)
        result = max(result, abs(a - b) / scale)
    return result


def validate(path: pathlib.Path, rtol: float, atol: float) -> int:
    rows = load_table(path)
    width = len(rows[0])
    failures: List[str] = []

    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row, start=1):
            if not math.isfinite(value):
                failures.append(f"row {i}, column {j}: non-finite value {value}")
        if row[0] <= 0.0:
            failures.append(f"row {i}: non-positive wavenumber {row[0]}")
        if i > 1 and row[0] <= rows[i - 2][0]:
            failures.append(f"row {i}: wavenumber grid is not strictly increasing")

    name = path.name
    if "matter_breakdown" in name and width >= 25:
        residuals = []
        for row in rows:
            checks = [
                (row[5], row[4] - row[2], "Delta_IR"),
                (row[15], row[13] + row[14], "P_ct_all"),
                (row[12], row[4] + row[8] + row[11], "P_after_loops_PNG"),
                (row[16], row[12] + row[15], "P_after_ct"),
                (row[18], row[16] + row[17], "P_total_EFT"),
            ]
            for lhs, expected, label in checks:
                residuals.append(lhs - expected)
                if abs(lhs - expected) > atol + rtol * max(abs(lhs), abs(expected), 1.0):
                    failures.append(f"k={row[0]:.6e}: {label} closure failed")
                    break
            if failures:
                break
        print(f"sum-rule max absolute residual: {max(map(abs, residuals)):.6e}")
    elif "tracer_real" in name and width >= 10:
        residuals = []
        for row in rows:
            expected = row[1] + row[2] + row[3] + row[6] + row[7] + row[8]
            residuals.append(row[9] - expected)
            if abs(row[9] - expected) > atol + rtol * max(abs(row[9]), abs(expected), 1.0):
                failures.append(
                    f"k={row[0]:.6e}: Ptotal != Ptree+P22+P13+PPNG+Pct+Pstoch"
                )
                break
        print(f"sum-rule max absolute residual: {max(map(abs, residuals)):.6e}")
    elif "tracer_real" in name and width >= 7:
        # Legacy 7-column layout.
        residuals = []
        for row in rows:
            expected = sum(row[1:6])
            residuals.append(row[6] - expected)
            if abs(row[6] - expected) > atol + rtol * max(abs(row[6]), abs(expected), 1.0):
                failures.append(
                    f"k={row[0]:.6e}: legacy Ptotal != Ptree+P22+P13+Pct+Pstoch"
                )
                break
        print(f"sum-rule max absolute residual: {max(map(abs, residuals)):.6e}")
    elif "pk_nl" in name and width >= 8:
        residuals = []
        for row in rows:
            expected = row[1] + row[2] + row[5]
            residuals.append(row[6] - expected)
            if abs(row[6] - expected) > atol + rtol * max(abs(row[6]), abs(expected), 1.0):
                failures.append(f"k={row[0]:.6e}: P_tree_plus_loop_PNG closure failed")
                break
        print(f"sum-rule max absolute residual: {max(map(abs, residuals)):.6e}")
    elif "pk_nl" in name and width >= 5:
        residuals = []
        for row in rows:
            expected = row[1] + row[2]
            residuals.append(row[3] - expected)
            if abs(row[3] - expected) > atol + rtol * max(abs(row[3]), abs(expected), 1.0):
                failures.append(f"k={row[0]:.6e}: legacy Pnl != Ptree+P1loop")
                break
        print(f"sum-rule max absolute residual: {max(map(abs, residuals)):.6e}")
    elif "multipoles" in name and width >= 7:
        if all(abs(row[3]) <= atol for row in rows) and any(abs(row[6]) > atol for row in rows):
            failures.append("total hexadecapole is identically zero while tree hexadecapole is not")
        print(
            "multipole maxima: "
            + ", ".join(
                f"P{ell}={max(abs(row[col]) for row in rows):.6e}"
                for ell, col in ((0, 1), (2, 2), (4, 3))
            )
        )

    print(f"checked {len(rows)} rows and {width} columns in {path}")
    if failures:
        for failure in failures[:20]:
            print(f"FAIL: {failure}", file=sys.stderr)
        if len(failures) > 20:
            print(f"FAIL: {len(failures)-20} further failures suppressed", file=sys.stderr)
        return 1
    print("PASS")
    return 0


def compare(
    path_a: pathlib.Path,
    path_b: pathlib.Path,
    column: int,
    kmin: float | None,
    kmax: float | None,
) -> int:
    rows_a = load_table(path_a)
    rows_b = load_table(path_b)
    if len(rows_a) != len(rows_b):
        print("FAIL: tables have different row counts", file=sys.stderr)
        return 1
    if column < 0 or column >= len(rows_a[0]) or column >= len(rows_b[0]):
        print("FAIL: requested zero-based column is outside one table", file=sys.stderr)
        return 1

    selected_a: List[float] = []
    selected_b: List[float] = []
    for row_a, row_b in zip(rows_a, rows_b):
        if not math.isclose(row_a[0], row_b[0], rel_tol=1.0e-10, abs_tol=1.0e-14):
            print("FAIL: k grids differ", file=sys.stderr)
            return 1
        k = row_a[0]
        if kmin is not None and k < kmin:
            continue
        if kmax is not None and k > kmax:
            continue
        selected_a.append(row_a[column])
        selected_b.append(row_b[column])
    if not selected_a:
        print("FAIL: no rows in requested k interval", file=sys.stderr)
        return 1

    rel = [
        abs(a - b) / max(abs(a), abs(b), 1.0e-300)
        for a, b in zip(selected_a, selected_b)
    ]
    rel_sorted = sorted(rel)
    median = rel_sorted[len(rel_sorted) // 2]
    p95 = rel_sorted[min(len(rel_sorted) - 1, int(0.95 * len(rel_sorted)))]
    print(f"compared {len(rel)} points, zero-based column {column}")
    print(f"relative difference: max={max(rel):.6e}, median={median:.6e}, p95={p95:.6e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=pathlib.Path)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), type=pathlib.Path)
    parser.add_argument("--column", type=int, default=2, help="zero-based comparison column")
    parser.add_argument("--kmin", type=float)
    parser.add_argument("--kmax", type=float)
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.column, args.kmin, args.kmax)
    if not args.files:
        parser.error("provide at least one output file or use --compare")
    status = 0
    for path in args.files:
        status |= validate(path, args.rtol, args.atol)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
