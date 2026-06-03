#!/usr/bin/env python3
import os
import glob
import csv
from pathlib import Path

import numpy as np
import tables as tb
import plotly.graph_objects as go
import streamlit as st

from invisible_cities.cities.components import baseline_subtractor
from invisible_cities.cities.components import build_pmap_dual_gain
from invisible_cities.cities.components import fourier_filter
from invisible_cities.cities.components import calibrate_fibers_hg
from invisible_cities.cities.components import calibrate_fibers_lg
from invisible_cities.cities.components import zero_suppress_wfs_hg
from invisible_cities.cities.components import zero_suppress_wfs_lg
from invisible_cities.core import system_of_units as units

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("ICTDIR", str(ROOT_DIR))
DATA_DIR = ROOT_DIR / "data"
SIPM_POSITIONS_CSV = ROOT_DIR / "scripts" / "hddemo_db_elecid_positions.csv"

# Candidate-selection defaults aligned with irene.conf.
S1_TMIN_US_DEFAULT, S1_TMAX_US_DEFAULT = 50.0, 250.0
S1_STRIDE_DEFAULT = 3
S1_LMIN_DEFAULT, S1_LMAX_DEFAULT = 1, 100
S1_REBIN_STRIDE_DEFAULT = 1

S2_TMIN_US_DEFAULT, S2_TMAX_US_DEFAULT = 248.0, 290.0
S2_STRIDE_DEFAULT = 2
S2_LMIN_DEFAULT, S2_LMAX_DEFAULT = 150, 100000
S2_REBIN_STRIDE_DEFAULT = 40

THR_SIPM_S2_DEFAULT = 1.5
PMT_SAMP_WID_NS_DEFAULT = 25.0
SIPM_SAMP_WID_US_DEFAULT = 1.0


def parse_run_number(path: str) -> int:
    name = Path(path).name
    parts = name.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return -1


def parse_chunk_number(path: str) -> int:
    name = Path(path).name
    parts = name.split("_")
    if len(parts) > 2 and parts[2].isdigit():
        return int(parts[2])
    return 10**9


def files_for_run(files, run_number: int):
    run_files = [f for f in files if parse_run_number(f) == int(run_number)]
    return sorted(run_files, key=lambda p: (parse_chunk_number(p), Path(p).name))


def resolve_event_location(run_files, global_event_idx: int):
    """Map a run-level event index to (file, local_event_index)."""
    if not run_files:
        return None, None, 0, 0, True

    remaining = int(max(0, global_event_idx))
    total_events = 0

    for f in run_files:
        n_events, _, _ = get_dataset_shape(f)
        total_events += int(n_events)
        if remaining < n_events:
            return f, remaining, total_events, remaining, False
        remaining -= int(n_events)

    # Requested event is beyond the run total; clamp to last available event.
    last_file = run_files[-1]
    last_n_events, _, _ = get_dataset_shape(last_file)
    local_idx = max(0, int(last_n_events) - 1)
    return last_file, local_idx, total_events, int(global_event_idx), True


def split_contiguous(indices: np.ndarray):
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, breaks)


def split_with_stride(indices: np.ndarray, stride: int):
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > stride)[0] + 1
    return np.split(indices, breaks)


def analyze_candidate(seg, t_us, sample_width_ns, tmin_us, tmax_us, lmin, lmax):
    t0 = float(t_us[seg[0]])
    t1 = float(t_us[seg[-1]] + sample_width_ns * 1e-3)
    width = int(seg[-1] + 1 - seg[0])

    reasons = []
    if t0 < tmin_us:
        reasons.append(f"starts before tmin ({t0:.3f} < {tmin_us:.3f} us)")
    if t1 > tmax_us:
        reasons.append(f"ends after tmax ({t1:.3f} > {tmax_us:.3f} us)")
    if not (lmin <= width <= lmax):
        reasons.append(f"length out of range ({width} not in [{lmin}, {lmax}] bins)")

    return {
        "segment": seg,
        "t0": t0,
        "t1": t1,
        "width_bins": width,
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }


def classify_candidate_segments(indices, stride, t_us, sample_width_ns, tmin_us, tmax_us, lmin, lmax):
    candidates = split_with_stride(np.asarray(indices, dtype=int), stride)
    analyzed, selected, rejected = [], [], []

    for seg in candidates:
        if len(seg) == 0:
            continue
        info = analyze_candidate(seg, t_us, sample_width_ns, tmin_us, tmax_us, lmin, lmax)
        analyzed.append(info)
        (selected if info["passed"] else rejected).append(seg)

    return analyzed, selected, rejected


def format_stage_a_block(label, analyzed):
    selected = sum(c["passed"] for c in analyzed)
    rejected = len(analyzed) - selected
    lines = [f"{label} candidates: {len(analyzed)} | selected: {selected} | rejected: {rejected}"]

    for i, c in enumerate(analyzed, 1):
        status = "SELECTED" if c["passed"] else "REJECTED"
        lines.append(
            f"  {label} {i:02d}: {c['t0']:8.3f}-{c['t1']:8.3f} us | "
            f"width={c['width_bins']:4d} bins | {status}"
        )
        if not c["passed"]:
            for reason in c["reasons"]:
                lines.append(f"         - {reason}")

    return lines


def build_stage_a_single_report(label, analyzed, n_in_pmap=None, pmap_error=None):
    lines = [f"=== Stage A ({label}): Candidate split + time/length selection ==="]
    lines.extend(format_stage_a_block(label, analyzed))
    lines.append("")
    lines.append("PMAP object built:")

    if n_in_pmap is not None:
        lines.append(f"  n{label} in pmap = {n_in_pmap}")
    else:
        lines.append(f"  n{label} in pmap = unavailable")
        if pmap_error:
            lines.append(f"  reason: {pmap_error}")

    return "\n".join(lines)


def build_stage_a_single_markdown(label, analyzed, n_in_pmap=None, pmap_error=None):
    selected = sum(c["passed"] for c in analyzed)
    rejected = len(analyzed) - selected

    lines = [
        f"**Summary**  ",
        f"Candidates: **{len(analyzed)}** | Selected: **{selected}** | Rejected: **{rejected}**",
        "",
        "**Candidate Details**",
    ]

    if not analyzed:
        lines.append("- No candidate regions found above threshold.")
    else:
        for i, c in enumerate(analyzed, 1):
            status = "Selected" if c["passed"] else "Rejected"
            lines.append(
                f"- **{label} {i:02d}**: {c['t0']:.3f}-{c['t1']:.3f} us | "
                f"width {c['width_bins']} bins | **{status}**"
            )
            if not c["passed"] and c["reasons"]:
                for reason in c["reasons"]:
                    lines.append(f"  - reason: {reason}")

    lines.append("")
    lines.append("**PMAP**")
    if n_in_pmap is not None:
        lines.append(f"- {label} peaks in PMAP: **{n_in_pmap}**")
    else:
        lines.append(f"- {label} peaks in PMAP: unavailable")
        if pmap_error:
            lines.append(f"- PMAP build note: {pmap_error}")

    return "\n".join(lines)


def build_stage_a_report(s1_analyzed, s2_analyzed, pmap_evt, pmap_error):
    lines = ["=== Stage A: Candidate split + time/length selection ==="]
    lines.extend(format_stage_a_block("S1", s1_analyzed))
    lines.extend(format_stage_a_block("S2", s2_analyzed))
    lines.append("")
    lines.append("PMAP object built:")

    if pmap_evt is not None:
        lines.append(f"  nS1 in pmap = {len(pmap_evt.s1s)}")
        lines.append(f"  nS2 in pmap = {len(pmap_evt.s2s)}")
    else:
        lines.append("  nS1 in pmap = unavailable")
        lines.append("  nS2 in pmap = unavailable")
        if pmap_error:
            lines.append(f"  reason: {pmap_error}")

    return "\n".join(lines)


def discover_waveform_files(data_dir: str):
    pattern = str(Path(data_dir) / "run_*_ldc1_trg0.waveforms.h5")
    return sorted(glob.glob(pattern))


@st.cache_data(show_spinner=False)
def get_dataset_shape(file_path: str):
    with tb.open_file(file_path, "r") as h5in:
        shape = h5in.root.RD.fiberrwf_hg.shape
    return shape


@st.cache_data(show_spinner=False)
def load_event(file_path: str, event_idx: int):
    with tb.open_file(file_path, "r") as h5in:
        fiber_hg = h5in.root.RD.fiberrwf_hg[event_idx]
        fiber_lg = h5in.root.RD.fiberrwf_lg[event_idx]
        sipm_wf = h5in.root.RD.sipmrwf[event_idx]
        sipm_sensors = h5in.root.Sensors.DataSiPM[:]
        event_no = int(h5in.root.Run.events[event_idx][0])
    return fiber_hg, fiber_lg, sipm_wf, sipm_sensors, event_no


@st.cache_data(show_spinner=False)
def load_sipm_positions(csv_path: str):
    out = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elecid = int(row["ElecID"])
            out[elecid] = (float(row["X"]), float(row["Y"]))
    return out


def get_s2_windows_us(pmap_evt, s2_selected, t_us, fiber_samp_wid_ns):
    windows = []

    if pmap_evt is not None and len(pmap_evt.s2s):
        try:
            for s2 in pmap_evt.s2s:
                times = np.asarray(s2.times, dtype=float)
                if len(times) == 0:
                    continue
                # PMAP times are in ns; convert to us.
                t0_us = float(times[0]) * 1e-3
                if len(times) > 1:
                    dt_us = float(np.median(np.diff(times))) * 1e-3
                else:
                    dt_us = float(fiber_samp_wid_ns) * 1e-3
                t1_us = float(times[-1]) * 1e-3 + dt_us
                windows.append((t0_us, t1_us))
            if windows:
                return windows
        except Exception:
            pass

    # Fallback to selected S2 candidate windows in fiber time.
    for seg in s2_selected:
        windows.append((float(t_us[seg[0]]), float(t_us[seg[-1]] + float(fiber_samp_wid_ns) * 1e-3)))
    return windows


def sipm_s2_charge_map_figure(sipm_wf_evt, sipm_sensors, positions_by_elecid, s2_windows_us, sipm_samp_wid_us):
    if not s2_windows_us:
        return None, 0

    n_samples = sipm_wf_evt.shape[1]
    t_sipm_us = np.arange(n_samples, dtype=float) * float(sipm_samp_wid_us)
    mask = np.zeros_like(t_sipm_us, dtype=bool)
    for t0, t1 in s2_windows_us:
        mask |= (t_sipm_us >= float(t0)) & (t_sipm_us <= float(t1))
    if not np.any(mask):
        return None, 0

    baseline_n = max(10, min(50, n_samples // 5))
    baseline = np.median(sipm_wf_evt[:, :baseline_n], axis=1)
    corrected = sipm_wf_evt - baseline[:, None]
    corrected = np.where(corrected > 0, corrected, 0.0)
    q = np.sum(corrected[:, mask], axis=1) * float(sipm_samp_wid_us)

    x_vals, y_vals, q_vals, labels = [], [], [], []
    for i in range(len(sipm_sensors)):
        elecid = int(sipm_sensors[i]["channel"])
        if elecid not in positions_by_elecid:
            continue
        x, y = positions_by_elecid[elecid]
        x_vals.append(x)
        y_vals.append(y)
        q_vals.append(float(q[i]))
        labels.append(elecid)

    if not x_vals:
        return None, 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=10,
                color=q_vals,
                colorscale="Turbo",
                colorbar=dict(title="Integrated charge"),
                line=dict(color="black", width=0.4),
            ),
            text=[f"ElecID {eid}<br>Q={qq:.2f}" for eid, qq in zip(labels, q_vals)],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="SiPM integrated charge in S2 valid window(s)",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_scaleanchor="x",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=520,
        font=dict(color="black"),
    )
    return fig, len(x_vals)


def overlay_plot(t_us, a, b, title, name_a, name_b, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=a, mode="lines", name=name_a, line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=t_us, y=b, mode="lines", name=name_b, line=dict(width=1.2)))
    fig.update_layout(
        title=title,
        xaxis_title="Time (us)",
        yaxis_title=y_title,
        height=350,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def threshold_plot(
    t_us,
    y,
    thr,
    thr_label,
    title,
    selected,
    rejected,
    selected_color,
    rejected_color,
    allowed_window=None,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_us, y=y, mode="lines", name="summed waveform", line=dict(width=1.2)))
    fig.add_hline(y=thr, line_dash="dash")

    if allowed_window is not None:
        t0, t1 = allowed_window
        fig.add_vrect(
            x0=float(t0),
            x1=float(t1),
            fillcolor="#a5d8ff",
            opacity=0.12,
            line_width=0,
        )

    for i, seg in enumerate(selected):
        fig.add_vrect(
            x0=float(t_us[seg[0]]),
            x1=float(t_us[seg[-1]]),
            fillcolor=selected_color,
            opacity=0.35,
            line_width=1,
            line_color=selected_color,
        )

    for i, seg in enumerate(rejected):
        fig.add_vrect(
            x0=float(t_us[seg[0]]),
            x1=float(t_us[seg[-1]]),
            fillcolor=rejected_color,
            opacity=0.28,
            line_width=1,
            line_color=rejected_color,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (us)",
        yaxis_title="pes",
        height=350,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def main():
    st.set_page_config(page_title="Irene Interactive Pipeline", layout="wide")
    st.title("Irene Interactive Pipeline")
    st.caption("Select run/event/channel and tune pipeline parameters live.")

    files = discover_waveform_files(str(DATA_DIR))
    if not files:
        st.error(f"No waveform files found in {DATA_DIR}")
        st.stop()

    available_runs = sorted({parse_run_number(f) for f in files if parse_run_number(f) >= 0})
    if not available_runs:
        st.error(f"No valid run files found in {DATA_DIR}")
        st.stop()

    default_run = 1128 if 1128 in available_runs else int(available_runs[0])

    with st.sidebar:
        st.header("Input")
        if st.button("Refresh run files"):
            st.cache_data.clear()
            st.rerun()

        run_number = st.number_input(
            "Run number",
            min_value=0,
            max_value=999999,
            value=int(default_run),
            step=1,
        )

        run_files = files_for_run(files, int(run_number))
        if not run_files:
            st.error(f"No files found for run {int(run_number)} in {DATA_DIR}")
            st.stop()

        chunk_count = sum(1 for f in run_files if parse_chunk_number(f) < 10**9)
        if chunk_count:
            st.caption(f"Available chunks for run {int(run_number)}: {chunk_count}")

        event_idx_requested = st.number_input(
            "Event index (run-level)",
            min_value=0,
            max_value=10000000,
            value=12,
            step=1,
        )

        selected_file, event_idx, total_events_in_run, _, event_clamped = resolve_event_location(
            run_files, int(event_idx_requested)
        )

        n_events, n_fibers, n_samples = get_dataset_shape(selected_file)
        file_name = Path(selected_file).name
        if event_clamped:
            st.warning(
                f"Requested event {int(event_idx_requested)} exceeds run total ({int(total_events_in_run) - 1} max). "
                f"Using last event from {file_name}."
            )
        else:
            st.caption(f"Using file: {file_name} | local event index: {int(event_idx)}")

        fiber_ch_requested = st.number_input(
            "Fiber channel",
            min_value=0,
            max_value=100000,
            value=4,
            step=1,
        )
        fiber_ch = min(int(fiber_ch_requested), int(n_fibers) - 1)
        if int(fiber_ch_requested) != fiber_ch:
            st.warning(f"Requested fiber channel {int(fiber_ch_requested)} exceeds max {int(n_fibers) - 1}. Using {fiber_ch}.")

        detector_db = st.selectbox("Detector DB", ["hddemojb", "hddemo"], index=0)

        st.header("Parameters")
        n_baseline = st.number_input("N_BASELINE", min_value=100, max_value=n_samples, value=2800, step=100)
        n_maw_s1 = st.number_input("N_MAW_S1", min_value=1, max_value=5000, value=10, step=1)
        n_maw_s2 = st.number_input("N_MAW_S2", min_value=1, max_value=5000, value=100, step=1)
        fiber_samp_wid = st.number_input("FIBER_SAMP_WID (ns)", min_value=1.0, max_value=1000.0, value=25.0, step=1.0)
        fiber_cutoff_mhz = st.number_input("FIBER_CUTOFF_FREQ_MHZ", min_value=0.1, max_value=100.0, value=3.0, step=0.1)
        thr_csum_s1 = st.number_input("THR_CSUM_S1 (pes)", min_value=0.0, max_value=1e6, value=100.0, step=1.0)
        thr_csum_s2 = st.number_input("THR_CSUM_S2 (pes)", min_value=0.0, max_value=1e6, value=60.0, step=1.0)

        st.subheader("S1 selection")
        s1_tmin_us = st.number_input("s1_tmin (us)", min_value=0.0, max_value=1e6, value=S1_TMIN_US_DEFAULT, step=1.0)
        s1_tmax_us = st.number_input("s1_tmax (us)", min_value=0.0, max_value=1e6, value=S1_TMAX_US_DEFAULT, step=1.0)
        s1_stride = st.number_input("s1_stride", min_value=1, max_value=10000, value=S1_STRIDE_DEFAULT, step=1)
        s1_lmin = st.number_input("s1_lmin", min_value=1, max_value=1000000, value=S1_LMIN_DEFAULT, step=1)
        s1_lmax = st.number_input("s1_lmax", min_value=1, max_value=1000000, value=S1_LMAX_DEFAULT, step=1)
        s1_rebin_stride = st.number_input("s1_rebin_stride", min_value=1, max_value=100000, value=S1_REBIN_STRIDE_DEFAULT, step=1)

        st.subheader("S2 selection")
        s2_tmin_us = st.number_input("s2_tmin (us)", min_value=0.0, max_value=1e6, value=S2_TMIN_US_DEFAULT, step=1.0)
        s2_tmax_us = st.number_input("s2_tmax (us)", min_value=0.0, max_value=1e6, value=S2_TMAX_US_DEFAULT, step=1.0)
        s2_stride = st.number_input("s2_stride", min_value=1, max_value=10000, value=S2_STRIDE_DEFAULT, step=1)
        s2_lmin = st.number_input("s2_lmin", min_value=1, max_value=1000000, value=S2_LMIN_DEFAULT, step=1)
        s2_lmax = st.number_input("s2_lmax", min_value=1, max_value=1000000, value=S2_LMAX_DEFAULT, step=1)
        s2_rebin_stride = st.number_input("s2_rebin_stride", min_value=1, max_value=100000, value=S2_REBIN_STRIDE_DEFAULT, step=1)

        st.subheader("PMAP sampling")
        thr_sipm_s2 = st.number_input("thr_sipm_s2 (pes)", min_value=0.0, max_value=1e6, value=THR_SIPM_S2_DEFAULT, step=0.1)
        pmt_samp_wid_ns = st.number_input("pmt_samp_wid (ns)", min_value=1.0, max_value=1000.0, value=PMT_SAMP_WID_NS_DEFAULT, step=1.0)
        sipm_samp_wid_us = st.number_input("sipm_samp_wid (us)", min_value=0.1, max_value=1000.0, value=SIPM_SAMP_WID_US_DEFAULT, step=0.1)

    st.info(
        f"Run {int(run_number)} | Requested event {int(event_idx_requested)} | "
        f"Using {Path(selected_file).name} (local event {int(event_idx)}/{n_events - 1}) | "
        f"Channel {fiber_ch}/{n_fibers - 1} | Samples {n_samples}"
    )

    try:
        fiber_hg_raw, fiber_lg_raw, sipm_wf_evt, sipm_sensors, event_number = load_event(selected_file, event_idx)
        positions_by_elecid = load_sipm_positions(str(SIPM_POSITIONS_CSV))

        t_us = np.arange(n_samples) * float(fiber_samp_wid) * 1e-3

        subtract_baseline = baseline_subtractor(int(n_baseline))
        bsfiber_hg = subtract_baseline(fiber_hg_raw)
        bsfiber_lg = subtract_baseline(fiber_lg_raw)

        apply_fourier_filter = fourier_filter(float(fiber_samp_wid), float(fiber_cutoff_mhz))
        bsffiber_hg = apply_fourier_filter(bsfiber_hg)
        bsffiber_lg = apply_fourier_filter(bsfiber_lg)

        cal_hg = calibrate_fibers_hg(detector_db, int(run_number), int(n_maw_s1))
        cal_lg = calibrate_fibers_lg(detector_db, int(run_number), int(n_maw_s2))
        cbsfiber_hg_maw, cbsfiber_hg_sum_maw = cal_hg(bsffiber_hg)
        cbsfiber_lg_maw, cbsfiber_lg_sum_maw = cal_lg(bsffiber_lg)

        zs_hg = zero_suppress_wfs_hg(float(thr_csum_s1))
        zs_lg = zero_suppress_wfs_lg(float(thr_csum_s2))
        s1_indices = zs_hg(cbsfiber_hg_sum_maw)
        s2_indices, s2_energies = zs_lg(cbsfiber_lg_sum_maw)

        s1_analyzed, s1_selected, s1_rejected = classify_candidate_segments(
            s1_indices,
            int(s1_stride),
            t_us,
            float(fiber_samp_wid),
            float(s1_tmin_us),
            float(s1_tmax_us),
            int(s1_lmin),
            int(s1_lmax),
        )
        s2_analyzed, s2_selected, s2_rejected = classify_candidate_segments(
            s2_indices,
            int(s2_stride),
            t_us,
            float(fiber_samp_wid),
            float(s2_tmin_us),
            float(s2_tmax_us),
            int(s2_lmin),
            int(s2_lmax),
        )

        pmap_evt = None
        pmap_error = None
        try:
            pmap_builder = build_pmap_dual_gain(
                detector_db,
                int(run_number),
                float(pmt_samp_wid_ns) * units.ns,
                float(sipm_samp_wid_us) * units.mus,
                int(s1_lmax),
                int(s1_lmin),
                int(s1_rebin_stride),
                int(s1_stride),
                float(s1_tmax_us) * units.mus,
                float(s1_tmin_us) * units.mus,
                int(s2_lmax),
                int(s2_lmin),
                int(s2_rebin_stride),
                int(s2_stride),
                float(s2_tmax_us) * units.mus,
                float(s2_tmin_us) * units.mus,
                float(thr_sipm_s2),
            )
            pmap_evt = pmap_builder(cbsfiber_hg_maw, cbsfiber_lg_maw, s1_indices, s2_indices, None)
        except Exception as e:
            pmap_error = str(e)

        if pmap_evt is not None:
            stage_a_s1_md = build_stage_a_single_markdown("S1", s1_analyzed, n_in_pmap=len(pmap_evt.s1s))
            stage_a_s2_md = build_stage_a_single_markdown("S2", s2_analyzed, n_in_pmap=len(pmap_evt.s2s))
        else:
            stage_a_s1_md = build_stage_a_single_markdown("S1", s1_analyzed, n_in_pmap=None, pmap_error=pmap_error)
            stage_a_s2_md = build_stage_a_single_markdown("S2", s2_analyzed, n_in_pmap=None, pmap_error=pmap_error)

        s2_windows_us = get_s2_windows_us(pmap_evt, s2_selected, t_us, float(fiber_samp_wid))
        sipm_map_fig, sipm_mapped = sipm_s2_charge_map_figure(
            sipm_wf_evt,
            sipm_sensors,
            positions_by_elecid,
            s2_windows_us,
            float(sipm_samp_wid_us),
        )

    except Exception as exc:
        st.error("Pipeline execution failed with current settings.")
        st.exception(exc)
        st.stop()

    st.subheader(f"Event number: {event_number}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            overlay_plot(
                t_us,
                fiber_hg_raw[fiber_ch],
                bsfiber_hg[fiber_ch],
                "HG: raw vs baseline-subtracted",
                "raw ADC",
                "baseline-subtracted ADC",
                "ADC",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            overlay_plot(
                t_us,
                fiber_lg_raw[fiber_ch],
                bsfiber_lg[fiber_ch],
                "LG: raw vs baseline-subtracted",
                "raw ADC",
                "baseline-subtracted ADC",
                "ADC",
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsfiber_hg[fiber_ch],
                bsffiber_hg[fiber_ch],
                "HG: before vs after Fourier filter",
                "before",
                "after",
                "ADC",
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsfiber_lg[fiber_ch],
                bsffiber_lg[fiber_ch],
                "LG: before vs after Fourier filter",
                "before",
                "after",
                "ADC",
            ),
            use_container_width=True,
        )

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsffiber_hg[fiber_ch],
                cbsfiber_hg_maw[fiber_ch],
                "HG: filtered ADC vs calibrated MAW",
                "filtered ADC",
                "calibrated MAW (pes)",
                "value",
            ),
            use_container_width=True,
        )
    with col6:
        st.plotly_chart(
            overlay_plot(
                t_us,
                bsffiber_lg[fiber_ch],
                cbsfiber_lg_maw[fiber_ch],
                "LG: filtered ADC vs calibrated MAW",
                "filtered ADC",
                "calibrated MAW (pes)",
                "value",
            ),
            use_container_width=True,
        )

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(
            threshold_plot(
                t_us,
                cbsfiber_hg_sum_maw,
                float(thr_csum_s1),
                f"S1 threshold = {thr_csum_s1:.1f}",
                "HG sum with S1 selected/rejected windows",
                s1_selected,
                s1_rejected,
                "#2faa60",
                "#d94a4a",
                allowed_window=(float(s1_tmin_us), float(s1_tmax_us)),
            ),
            use_container_width=True,
        )
    with col8:
        st.plotly_chart(
            threshold_plot(
                t_us,
                cbsfiber_lg_sum_maw,
                float(thr_csum_s2),
                f"S2 threshold = {thr_csum_s2:.1f}",
                "LG sum with S2 selected/rejected windows",
                s2_selected,
                s2_rejected,
                "#1e8e5a",
                "#c73e3e",
                allowed_window=(float(s2_tmin_us), float(s2_tmax_us)),
            ),
            use_container_width=True,
        )

    with st.expander("Stage A candidate diagnostic", expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown("### S1 diagnostics")
            st.markdown(stage_a_s1_md)
        with right:
            st.markdown("### S2 diagnostics")
            st.markdown(stage_a_s2_md)

    st.subheader("SiPM S2 Charge Map")
    if sipm_map_fig is None:
        st.info("No S2 window available for SiPM integration with current settings.")
    else:
        st.caption(f"Mapped SiPM sensors: {sipm_mapped}")
        st.plotly_chart(sipm_map_fig, use_container_width=True)

    with st.expander("Current configuration"):
        st.json(
            {
                "file": selected_file,
                "run_number": int(run_number),
                "event_idx_requested": int(event_idx_requested),
                "event_idx_local": int(event_idx),
                "n_events_file": int(n_events),
                "n_events_run_total": int(total_events_in_run),
                "detector_db": detector_db,
                "event_idx": int(event_idx),
                "fiber_ch": int(fiber_ch),
                "N_BASELINE": int(n_baseline),
                "N_MAW_S1": int(n_maw_s1),
                "N_MAW_S2": int(n_maw_s2),
                "FIBER_SAMP_WID": float(fiber_samp_wid),
                "FIBER_CUTOFF_FREQ_MHZ": float(fiber_cutoff_mhz),
                "THR_CSUM_S1": float(thr_csum_s1),
                "THR_CSUM_S2": float(thr_csum_s2),
                "s1_tmin_us": float(s1_tmin_us),
                "s1_tmax_us": float(s1_tmax_us),
                "s1_stride": int(s1_stride),
                "s1_lmin": int(s1_lmin),
                "s1_lmax": int(s1_lmax),
                "s1_rebin_stride": int(s1_rebin_stride),
                "s2_tmin_us": float(s2_tmin_us),
                "s2_tmax_us": float(s2_tmax_us),
                "s2_stride": int(s2_stride),
                "s2_lmin": int(s2_lmin),
                "s2_lmax": int(s2_lmax),
                "s2_rebin_stride": int(s2_rebin_stride),
                "thr_sipm_s2": float(thr_sipm_s2),
                "pmt_samp_wid_ns": float(pmt_samp_wid_ns),
                "sipm_samp_wid_us": float(sipm_samp_wid_us),
            }
        )


if __name__ == "__main__":
    main()
