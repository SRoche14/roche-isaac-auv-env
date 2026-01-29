#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path

NUM_FIELDS = {
    "successes", "failures", "success_rate", "avg_energy",
    "stop_n", "stop_min", "stop_q1", "stop_median", "stop_q3", "stop_max",
    "stop_iqr", "stop_whis_low", "stop_whis_high", "stop_mean", "stop_std",
    "stop_outliers_count",
}

PARAM_FIELDS = [
    "task", "checkpoint", "goal_o", "starting_o", "vel", "offset",
    "num_envs", "eval_dt", "timeout_s", "success_radius_m",
]

def to_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compare_rows(r1, r2):
    out = {}
    for k in NUM_FIELDS:
        v1 = to_float(r1.get(k))
        v2 = to_float(r2.get(k))
        if v1 is None or v2 is None:
            out[f"{k}_delta"] = ""
            out[f"{k}_pct_delta"] = ""
            continue
        delta = v2 - v1
        out[f"{k}_delta"] = delta
        out[f"{k}_pct_delta"] = (100.0 * delta / v1) if v1 != 0 else ""
    return out

def main():
    p = argparse.ArgumentParser(description="Compare two summary.csv files line-by-line.")
    p.add_argument("file1", type=Path, help="Baseline summary.csv")
    p.add_argument("file2", type=Path, help="New summary.csv")
    p.add_argument("-o", "--out", type=Path, default=Path("summary_compare.csv"))
    p.add_argument(
        "--show",
        default="success_rate,avg_energy,stop_median,stop_mean,stop_iqr,stop_max",
        help="Metrics to highlight in console summary",
    )
    p.add_argument("--top", type=int, default=5, help="Top-N rows by absolute delta to print")
    args = p.parse_args()

    with args.file1.open(newline="") as f1, args.file2.open(newline="") as f2:
        r1 = list(csv.DictReader(f1))
        r2 = list(csv.DictReader(f2))

    if len(r1) != len(r2):
        raise SystemExit(f"Length mismatch: {len(r1)} vs {len(r2)}")

    delta_fields = []
    for k in NUM_FIELDS:
        delta_fields.extend([f"{k}_delta", f"{k}_pct_delta"])
    out_fields = ["row_index", "timestamp_1", "timestamp_2"] + PARAM_FIELDS + list(NUM_FIELDS) + delta_fields

    rows_out = []
    highlight = [s.strip() for s in args.show.split(",") if s.strip()]
    # Track sums and valid-row counts per metric
    agg = {m: {"sum_delta": 0.0, "valid": 0} for m in highlight}

    for i, (a, b) in enumerate(zip(r1, r2)):
        deltas = compare_rows(a, b)

        # Aggregate only when delta is numeric (both files had values)
        for m in highlight:
            d = deltas.get(f"{m}_delta")
            if isinstance(d, (int, float)):
                agg[m]["sum_delta"] += d
                agg[m]["valid"] += 1

        base = {
            "row_index": i,
            "timestamp_1": a.get("timestamp", ""),
            "timestamp_2": b.get("timestamp", ""),
        }
        for k in PARAM_FIELDS:
            base[k] = b.get(k, a.get(k, ""))
        for k in NUM_FIELDS:
            base[k] = b.get(k, "")
        row = {**base, **deltas}
        rows_out.append(row)

    with args.out.open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=out_fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Wrote: {args.out}")
    print("== Aggregate delta (file2 - file1) over valid rows ==")
    for m in highlight:
        s = agg[m]["sum_delta"]
        v = agg[m]["valid"]
        avg = (s / v) if v > 0 else ""
        if v > 0:
            print(f"- {m}: total Δ = {s:+.6g} | avg Δ = {avg:+.6g} over {v} valid rows")
        else:
            print(f"- {m}: no valid rows")

    for m in highlight:
        key = f"{m}_delta"
        scored = []
        for r in rows_out:
            d = r.get(key)
            if isinstance(d, (int, float)):
                scored.append((abs(d), r["row_index"], d))
        scored.sort(reverse=True)
        if not scored:
            continue
        print(f"\nTop {min(args.top, len(scored))} by |Δ{m}|:")
        for absd, idx, d in scored[:args.top]:
            ro = rows_out[idx]
            sig = f"goal_o={ro.get('goal_o')} starting_o={ro.get('starting_o')} offset={ro.get('offset')} vel={ro.get('vel')}"
            print(f"  row {idx}: Δ{m}={d:+.6g} | {sig}")

if __name__ == "__main__":
    main()
