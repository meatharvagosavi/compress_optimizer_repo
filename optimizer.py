"""
Interactive Jupyter notebook style module: compress_optimizer_notebook.py

Features (notebook-friendly):
- Load setups from CSV or use demo values
- Interactive widgets: budget slider, SSIM slider, choose CSV upload or demo
- Compute lower convex hull and optimal mix per time budget
- Plot hull, show plan, simulate allocation across N files
- PDF path to the paper included for reference

Paper file (uploaded): /mnt/data/algorithms-18-00135.pdf

Usage: open this file in a Jupyter/Colab notebook cell using ``%run compress_optimizer_notebook.py``
or import functions from it. The bottom of this file builds an interactive UI using ipywidgets.

"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import io
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import ipywidgets as widgets

PAPER_PATH = '/mnt/data/algorithms-18-00135.pdf'

@dataclass
class Setup:
    name: str
    time_ms: float
    size_bytes: float
    quality: Optional[float] = None
    ssim: Optional[float] = None
    def as_point(self) -> Tuple[float, float]:
        return (self.time_ms, self.size_bytes

def preprocess_setups(setups: List[Setup]) -> List[Setup]:
    time_map: Dict[float, Setup] = {}
    for s in setups:
        t = s.time_ms
        if (t not in time_map) or (s.size_bytes < time_map[t].size_bytes):
            time_map[t] = s
    cleaned = list(time_map.values())
    cleaned.sort(key=lambda x: x.time_ms)
    return cleaned

def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def lower_convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    return lower

def useful_setups_from_list(setups: List[Setup], ssim_min: Optional[float] = None) -> List[Setup]:
    if ssim_min is not None:
        filtered = [s for s in setups if (s.ssim is not None and s.ssim >= ssim_min)]
    else:
        filtered = setups[:]
    cleaned = preprocess_setups(filtered)
    points = [s.as_point() for s in cleaned]
    hull_points = lower_convex_hull(points)
    hull_setups: List[Setup] = []
    for (t, b) in hull_points:
        for s in cleaned:
            if math.isclose(s.time_ms, t, rel_tol=1e-9, abs_tol=1e-9) and math.isclose(s.size_bytes, b, rel_tol=1e-9, abs_tol=1e-9):
                hull_setups.append(s)
                break
    hull_setups.sort(key=lambda x: x.time_ms)
    return hull_setups

def find_optimal_mix_single_dataset(hull: List[Setup], time_budget_ms: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not hull:
        result['status'] = 'NO_SETUP'
        result['reason'] = 'empty hull'
        return result
    hull_sorted = sorted(hull, key=lambda x: x.time_ms)
    times = [s.time_ms for s in hull_sorted]
    if time_budget_ms < times[0] - 1e-9:
        result['status'] = 'INFEASIBLE'
        result['reason'] = 'time budget too small'
        return result
    if time_budget_ms >= times[-1] - 1e-9:
        result['status'] = 'SINGLE'
        result['plan'] = (hull_sorted[-1], 1.0)
        return result
    for i in range(len(hull_sorted) - 1):
        s_fast = hull_sorted[i]
        s_slow = hull_sorted[i + 1]
        tb, ta = s_fast.time_ms, s_slow.time_ms
        if tb - 1e-12 <= time_budget_ms <= ta + 1e-12:
            if math.isclose(time_budget_ms, tb, rel_tol=1e-12, abs_tol=1e-12):
                result['status'] = 'SINGLE'
                result['plan'] = (s_fast, 1.0)
                return result
            if math.isclose(time_budget_ms, ta, rel_tol=1e-12, abs_tol=1e-12):
                result['status'] = 'SINGLE'
                result['plan'] = (s_slow, 1.0)
                return result
            ra = (time_budget_ms - tb) / (ta - tb)
            ra = max(0.0, min(1.0, ra))
            result['status'] = 'MIX'
            result['mix'] = (s_slow, s_fast, ra)
            result['plan'] = {s_slow.name: ra, s_fast.name: 1.0 - ra}
            return result
    result['status'] = 'ERROR'
    result['reason'] = 'no adjacent hull segment found'
    return result

def example_setups_from_paper() -> List[Setup]:
    return [
        Setup("Gzip", 1784.0, 7700660.0),
        Setup("ArithmeticCoding", 7928.0, 13906174.0),
        Setup("Bzip2", 16514.0, 7216597.0),
        Setup("XZ", 25974.0, 6793836.0)
    ]

def setups_from_csv_bytes(csv_bytes: bytes) -> List[Setup]:
    text = csv_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(text), header=None)
 
    setups: List[Setup] = []
    for idx, row in df.iterrows():
        try:
            name = str(row[0])
            t = float(row[1])
            b = float(row[2])
            q = float(row[3]) if len(row) > 3 and not pd.isna(row[3]) else None
            s = float(row[4]) if len(row) > 4 and not pd.isna(row[4]) else None
            setups.append(Setup(name, t, b, q, s))
        except Exception as e:
            continue
    return setups

def plot_hull(hull: List[Setup], time_budget_ms: Optional[float] = None):
    if not hull:
        print("No hull to plot")
        return
    times = [s.time_ms for s in hull]
    sizes = [s.size_bytes for s in hull]
    plt.figure(figsize=(8,4.8))
    plt.plot(times, sizes, marker='o')
    for s in hull:
        plt.annotate(s.name, (s.time_ms, s.size_bytes), textcoords='offset points', xytext=(6,-8))
    if time_budget_ms is not None:
        plt.axvline(x=time_budget_ms, linestyle='--')
        res = find_optimal_mix_single_dataset(hull, time_budget_ms)
        if res['status'] == 'SINGLE':
            s, _ = res['plan']
            plt.scatter([s.time_ms], [s.size_bytes], s=100, marker='s')
        elif res['status'] == 'MIX':
            s_slow, s_fast, ra = res['mix']
            mixed_size = ra * s_slow.size_bytes + (1-ra) * s_fast.size_bytes
            plt.scatter([time_budget_ms], [mixed_size], s=100, marker='X')
    plt.xlabel('Time (ms)')
    plt.ylabel('Compressed size (bytes)')
    plt.title('Lower convex hull: time vs compressed size')
    plt.grid(alpha=0.3)
    plt.show()

def build_interactive_ui():
    title = widgets.HTML(value=f"<h3>Compressor optimizer (paper: <code>{PAPER_PATH}</code>)</h3>")

    upload = widgets.FileUpload(accept='.csv', multiple=False, description='Upload CSV')
    demo_btn = widgets.Button(description='Use demo data', button_style='info')
    budget_slider = widgets.FloatSlider(value=22000.0, min=0.0, max=60000.0, step=100.0, description='Budget (ms)')
    ssim_slider = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='SSIM min')
    simulate_files = widgets.IntText(value=10, description='Simulate N files')
    run_btn = widgets.Button(description='Compute plan', button_style='primary')
    plot_chk = widgets.Checkbox(value=True, description='Show plot')
    out = widgets.Output()

    state = {'setups': example_setups_from_paper()}

    def on_demo_clicked(b):
        state['setups'] = example_setups_from_paper()
        with out:
            out.clear_output()
            print('Demo data loaded (paper example).')

    def on_upload_change(change):
        if upload.value:
            key = next(iter(upload.value))
            content = upload.value[key]['content']
            try:
                setups = setups_from_csv_bytes(content)
                if not setups:
                    with out:
                        out.clear_output()
                        print('No valid rows found in CSV. Expect: name,time_ms,size_bytes[,quality,ssim]')
                    return
                state['setups'] = setups
                with out:
                    out.clear_output()
                    print(f'Loaded {len(setups)} setups from uploaded CSV.')
            except Exception as e:
                with out:
                    out.clear_output()
                    print('Failed to parse CSV:', e)

    def on_run(b):
        with out:
            out.clear_output()
            setups = state.get('setups', [])
            print(f'Using {len(setups)} setups')
            ssim_min = ssim_slider.value if ssim_slider.value > 0 else None
            hull = useful_setups_from_list(setups, ssim_min=ssim_min)
            if not hull:
                print('No useful setups after filtering. Try lowering SSIM or checking data.')
                return
            print('Lower hull setups:')
            for s in hull:
                print(f' - {s.name} | time={s.time_ms} ms | size={s.size_bytes} bytes | ssim={s.ssim}')
            budget = budget_slider.value
            plan = find_optimal_mix_single_dataset(hull, budget)
            print('\nTime budget:', budget, 'ms ->', plan['status'])
            if plan['status'] == 'SINGLE':
                s, _ = plan['plan']
                print(f'Use single setup: {s.name} (time {s.time_ms} ms, size {s.size_bytes} bytes)')
            elif plan['status'] == 'MIX':
                s_slow, s_fast, ra = plan['mix']
                print(f'MIX: {s_slow.name} fraction {ra*100:.2f}% + {s_fast.name} fraction {(1-ra)*100:.2f}%')
                print(f' {s_slow.name}: time {s_slow.time_ms}, size {s_slow.size_bytes}')
                print(f' {s_fast.name}: time {s_fast.time_ms}, size {s_fast.size_bytes}')
            else:
                print('No feasible plan:', plan.get('reason',''))
      
            N = simulate_files.value if simulate_files.value and simulate_files.value > 0 else 0
            if N > 0 and plan['status'] in ('MIX','SINGLE'):
                sim = simulate_allocation_for_files(hull, budget, [1.0]*N)
                print('\nSimulation across', N, 'files:')
                print(sim)
            if plot_chk.value:
                plot_hull(hull, time_budget_ms=budget)

    demo_btn.on_click(on_demo_clicked)
    upload.observe(on_upload_change, names='value')
    run_btn.on_click(on_run)

    controls_row1 = widgets.HBox([upload, demo_btn, plot_chk])
    controls_row2 = widgets.HBox([budget_slider, ssim_slider, simulate_files])
    controls_row3 = widgets.HBox([run_btn])
    paper_info = widgets.HTML(value=f"<p style='font-size:90%'>Paper (uploaded): <code>{PAPER_PATH}</code></p>")
    ui = widgets.VBox([title, controls_row1, controls_row2, controls_row3, out, paper_info])
    display(ui)

def simulate_allocation_for_files(hull: List[Setup], time_budget_ms: float, file_times: List[float]) -> Dict[str, Any]:
    N = len(file_times)
    total_files_weight = sum(file_times)
    res_single = find_optimal_mix_single_dataset(hull, time_budget_ms)
    assignment: Dict[str, float] = {}
    if res_single['status'] == 'INFEASIBLE':
        return {'status': 'INFEASIBLE', 'reason': res_single.get('reason', '')}
    if res_single['status'] == 'SINGLE':
        s, frac = res_single['plan']
        assignment[s.name] = 1.0
        return {'status': 'SINGLE', 'assignment': assignment, 'plan': res_single}
    if res_single['status'] == 'MIX':
        s_slow, s_fast, ra = res_single['mix']
        desired_slow_weight = ra * total_files_weight
        files_sorted = sorted([(w, idx) for idx, w in enumerate(file_times)], key=lambda x: x[0])
        assigned_slow_weight = 0.0
        assigned_fast_weight = 0.0
        slow_files = []
        fast_files = []
        for w, idx in reversed(files_sorted):
            if assigned_slow_weight + w <= desired_slow_weight + 1e-12:
                slow_files.append(idx)
                assigned_slow_weight += w
            else:
                fast_files.append(idx)
                assigned_fast_weight += w
        split_info = None
        shortfall = desired_slow_weight - assigned_slow_weight
        if abs(shortfall) > 1e-12 and fast_files:
            smallest_fast = min(fast_files, key=lambda i: file_times[i])
            w = file_times[smallest_fast]
            f = shortfall / w
            f = max(0.0, min(1.0, f))
            split_info = {'file_index': smallest_fast, 'to_slow_fraction': f}
            assigned_slow_weight += f * w
            assigned_fast_weight -= f * w
        assignment = {
            s_slow.name: assigned_slow_weight / total_files_weight if total_files_weight > 0 else 0.0,
            s_fast.name: assigned_fast_weight / total_files_weight if total_files_weight > 0 else 0.0
        }
        return {
            'status': 'MIX_FILES',
            'assignment': assignment,
            'slow_files': slow_files,
            'fast_files': fast_files,
            'split_info': split_info,
            'plan': res_single
        }
    return {'status': 'ERROR', 'reason': 'unexpected'}


try:
    build_interactive_ui()
except Exception as e:
    print('Widget UI could not be displayed (are you in a compatible notebook environment?).')
    print('Error:', e)
    print('You can still import functions from this module and call them programmatically.')
