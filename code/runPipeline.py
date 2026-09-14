#!/usr/bin/env python3
"""
runPipeline.py — run the full three-stage synthesis for many areas.

Stages, in order, per area:

    generateIndividuals.py  ->  generateHouseholds.py  ->  assignHouseholds.py

Assignment consumes the tensors written by the two generation stages, so the stages
are strictly ordered *within* an area. Areas, however, are independent: each is
synthesised from its own published margins and writes to its own output directory,
so parallelism is across areas, never across stages. `--workers N` runs N areas
concurrently, each executing its three stages in sequence.

Examples
--------
    # all 983 Greater London areas; GPUs detected and used automatically
    python code/runPipeline.py --region london

    # three GPUs, three areas on each -- small models leave a big card idle otherwise
    python code/runPipeline.py --region london --per-gpu 3

    # resume an interrupted sweep
    python code/runPipeline.py --region london --per-gpu 3 --skip-existing

    # one area at a time on one card: the only way to get clean per-area timings
    python code/runPipeline.py --region oxford --gpus 0 --workers 1

    # five-area representative Greater London subset (paper table)
    python code/runPipeline.py --region subset

    # one stage only, for a named subset
    python code/runPipeline.py --areas E02000800,E02000123 --stages assignment

GPU allocation
--------------
GPUs are detected automatically (`--gpus auto`, the default) by asking torch in a
throwaway subprocess. Worker slots are dealt round-robin across the cards found, and
`CUDA_VISIBLE_DEVICES` is set per child, so each stage script -- all of which ask for
`cuda` generically -- lands on its assigned card and sees it as cuda:0. Without this
every process would default to device 0 and a multi-GPU box would run at 1/N.

A worker leases a slot for a whole area and returns it at the end, so all three stages
of an area stay on one card and a worker that finishes early picks up the next area on
whichever card just came free. That self-balances; a static area-to-GPU mapping would
let one card fall behind whenever areas differ in cost, which they do.

Default worker count is (GPUs found x --per-gpu), or 1 on CPU. Override the count with
--workers, the device list with --gpus 0,2, or force CPU with --gpus none.

Collected artefacts
-------------------
Each stage leaves its working files under code/outputs/<stage>_<area>/ as before.
On top of that, the primary tensor each stage produces can be copied into a single
release tree, one directory per area:

    GreaterLondonSyntheticPopulation/
        E02000001/
            E02000001_person_nodes.pt          <- individuals
            E02000001_household_nodes.pt       <- households
            E02000001_final_assignments.pt     <- assignment
        E02000002/
            ...
        manifest.csv

Copying happens for skipped stages too, so re-running a finished sweep with
--skip-existing fills the tree without retraining anything. Change the location with
--collect-dir, or turn it off with --no-collect.

A note on timings
-----------------
Per-area durations recorded with --workers > 1 include time the process spent waiting
while other areas held the GPU, so they are NOT the cost of synthesising one area.
Use --workers 1 to measure per-area cost, and report total wall-clock for a parallel
sweep separately, stating the concurrency. The summary CSV records the `workers`
setting on every row so the two can never be conflated after the fact.
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))

STAGE_SCRIPTS = {
    'individuals': 'generateIndividuals.py',
    'households': 'generateHouseholds.py',
    'assignment': 'assignHouseholds.py',
}
STAGE_ORDER = ['individuals', 'households', 'assignment']

# Output file whose existence means the stage completed for that area.
# Assignment in this project writes under assignment_hp_tuning_{area}/.
STAGE_SENTINEL = {
    'individuals': os.path.join('outputs', 'individuals_{area}', 'person_nodes.pt'),
    'households': os.path.join('outputs', 'households_{area}', 'household_nodes.pt'),
    'assignment': os.path.join('outputs', 'assignment_hp_tuning_{area}', 'final_assignments.pt'),
}

# Primary artefact per stage copied into the collected population directory.
STAGE_ARTIFACT = {
    'individuals': (os.path.join('outputs', 'individuals_{area}', 'person_nodes.pt'),
                    '{area}_person_nodes.pt'),
    'households': (os.path.join('outputs', 'households_{area}', 'household_nodes.pt'),
                   '{area}_household_nodes.pt'),
    'assignment': (os.path.join('outputs', 'assignment_hp_tuning_{area}', 'final_assignments.pt'),
                   '{area}_final_assignments.pt'),
}

OXFORD_AREAS = [
    'E02005940', 'E02005941', 'E02005942', 'E02005943', 'E02005944', 'E02005945',
    'E02005946', 'E02005947', 'E02005948', 'E02005949', 'E02005950', 'E02005951',
    'E02005953', 'E02005954', 'E02005955', 'E02005956', 'E02005957',
]

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def detect_gpus():
    """Return a list of visible CUDA device indices, or [] if there are none.

    Asked of torch in a throwaway subprocess rather than by importing torch here: the
    runner itself never needs CUDA, and initialising it in the parent would put a
    context on device 0 that every child then inherits around.
    """
    probe = ('import torch,sys; '
             'sys.stdout.write(str(torch.cuda.device_count()) if torch.cuda.is_available() else "0")')
    try:
        out = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                             text=True, timeout=120)
        return list(range(int(out.stdout.strip() or 0)))
    except Exception:
        return []


def describe_gpus(devices):
    """One line per device: index, name, VRAM. Best-effort; returns [] on failure."""
    probe = ('import torch\n'
             'for i in range(torch.cuda.device_count()):\n'
             '    p = torch.cuda.get_device_properties(i)\n'
             '    print("%d\\t%s\\t%.0f" % (i, p.name, p.total_memory / 1024**3))\n')
    try:
        out = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                             text=True, timeout=120)
        lines = []
        for ln in out.stdout.strip().splitlines():
            idx, name, vram = ln.split('\t')
            if int(idx) in devices:
                lines.append(f'  GPU {idx}: {name} ({vram} GB)')
        return lines
    except Exception:
        return []


def resolve_gpus(args):
    """Decide which GPU each worker slot uses. Returns a list of device ids, or [].

    --gpus auto   (default) ask torch what is present
    --gpus 0,2    use exactly these devices
    --gpus none   force CPU for every worker
    """
    spec = (args.gpus or 'auto').strip().lower()
    if spec in ('none', 'cpu', ''):
        return []
    if spec == 'auto':
        return detect_gpus()
    try:
        return [int(x) for x in spec.split(',') if x.strip() != '']
    except ValueError:
        raise SystemExit(f'--gpus expects "auto", "none", or a comma-separated list; got {args.gpus!r}')


def load_london_areas():
    """Read the 983 Greater London MSOA codes from Utils/greaterLondonAreas.py."""
    path = os.path.join(HERE, 'Utils', 'greaterLondonAreas.py')
    if not os.path.exists(path):
        raise SystemExit(f'cannot find {path}')
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    codes = list(dict.fromkeys(re.findall(r"'(E0[0-9]{7})'", text)))
    if not codes:
        raise SystemExit('no area codes found in greaterLondonAreas.py')
    return codes


def load_representative_subset():
    """Load REPRESENTATIVE_SUBSET (5 paper areas) from Utils/greaterLondonAreas.py."""
    path = os.path.join(HERE, 'Utils', 'greaterLondonAreas.py')
    if not os.path.exists(path):
        raise SystemExit(f'cannot find {path}')
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    match = re.search(r'REPRESENTATIVE_SUBSET\s*=\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        raise SystemExit('REPRESENTATIVE_SUBSET not found in greaterLondonAreas.py')
    codes = re.findall(r"'(E0[0-9]{7})'", match.group(1))
    if not codes:
        raise SystemExit('REPRESENTATIVE_SUBSET is empty in greaterLondonAreas.py')
    return codes


def resolve_areas(args):
    # Explicit area lists take precedence over --region, which has a default
    # ('london') and would otherwise always win.
    if args.areas_file:
        with open(args.areas_file, encoding='utf-8') as fh:
            areas = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
    elif args.areas:
        areas = [a.strip() for a in args.areas.split(',') if a.strip()]
    elif args.region == 'oxford':
        areas = list(OXFORD_AREAS)
    elif args.region == 'subset':
        areas = load_representative_subset()
    else:
        areas = load_london_areas()
    if args.limit:
        areas = areas[:args.limit]
    return areas


def stage_done(area, stage):
    return os.path.exists(os.path.join(HERE, STAGE_SENTINEL[stage].format(area=area)))


def build_command(stage, area, args):
    # SP_GNN stage scripts only accept --area_code (HP grids are hardcoded).
    return [sys.executable, '-u', STAGE_SCRIPTS[stage], '--area_code', area]


def child_env(args, device=None):
    env = os.environ.copy()
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    # Each PyTorch process otherwise claims many intra-op threads; with several workers
    # that oversubscribes the CPU and slows every worker down. Pin it explicitly.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        env[var] = str(args.threads_per_worker)
    # The stage scripts all ask for `cuda` generically, so the device is chosen here by
    # masking. The child sees its assigned card as cuda:0 whichever physical one it is.
    if device is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
    return env


def artifact_bytes(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return ''


def collect_stage(area, stage, args):
    """Copy this stage's primary tensor into the collected population directory.

    Runs for completed *and* skipped stages, so re-running a finished sweep with
    --skip-existing populates the collection without retraining anything.
    Returns (dest_path, size_bytes); dest_path is '' if there was nothing to copy.
    """
    if args.no_collect:
        return '', ''
    src_tpl, dest_tpl = STAGE_ARTIFACT[stage]
    src = os.path.join(HERE, src_tpl.format(area=area))
    if not os.path.exists(src):
        log(f'  [{area}] {stage}: no artefact to collect at {os.path.relpath(src, HERE)}')
        return '', ''
    dest_dir = os.path.join(args.collect_dir, area)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, dest_tpl.format(area=area))
    shutil.copyfile(src, dest)
    return dest, artifact_bytes(dest)


def run_stage(area, stage, args, log_dir, device=None):
    """Run one stage for one area. Returns (status, seconds, logfile)."""
    if args.skip_existing and stage_done(area, stage):
        return 'skipped', 0.0, ''

    cmd = build_command(stage, area, args)
    log_path = os.path.join(log_dir, f'{area}_{stage}.log')
    started = time.time()

    if args.dry_run:
        where = f'GPU {device}' if device is not None else 'CPU'
        log(f'  [dry-run] {area} {stage} on {where}: {" ".join(cmd)}')
        return 'dry-run', 0.0, log_path

    try:
        with open(log_path, 'w', encoding='utf-8', errors='replace') as fh:
            proc = subprocess.run(
                cmd, cwd=HERE, env=child_env(args, device), stdout=fh,
                stderr=subprocess.STDOUT, timeout=args.timeout or None,
            )
        elapsed = time.time() - started
        if proc.returncode == 0:
            return 'ok', elapsed, log_path
        return f'failed(rc={proc.returncode})', elapsed, log_path
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - started, log_path
    except Exception as exc:
        return f'error({type(exc).__name__})', time.time() - started, log_path


def write_manifest(collect_dir, run_rows):
    """Rebuild manifest.csv from what is actually on disk in the collection directory."""
    if not os.path.isdir(collect_dir):
        return None
    known = {(r['area_code'], r['artifact']): r['bytes']
             for r in run_rows if r.get('artifact') and r.get('bytes') != ''}

    entries = []
    for area in sorted(os.listdir(collect_dir)):
        area_dir = os.path.join(collect_dir, area)
        if not os.path.isdir(area_dir):
            continue
        for stage in STAGE_ORDER:
            fname = STAGE_ARTIFACT[stage][1].format(area=area)
            path = os.path.join(area_dir, fname)
            if not os.path.exists(path):
                continue
            size = known.get((area, fname))
            if size is None:
                size = artifact_bytes(path)
            entries.append({
                'area_code': area, 'stage': stage, 'file': f'{area}/{fname}',
                'bytes': size,
            })
    manifest_path = os.path.join(collect_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['area_code', 'stage', 'file', 'bytes'])
        writer.writeheader()
        writer.writerows(entries)

    areas = len({e['area_code'] for e in entries})
    complete = sum(1 for area in {e['area_code'] for e in entries}
                   if sum(1 for e in entries if e['area_code'] == area) == len(STAGE_ORDER))
    print(f'\nCollected population: {len(entries)} artefacts across {areas} areas '
          f'({complete} with all three stages) in {collect_dir}')
    print(f'Manifest written to {manifest_path}')
    return manifest_path


def run_area(area, args, log_dir, counter, slots=None):
    """Run all requested stages for one area, in order. Stops at the first failure.

    `slots` is a queue of device ids (one entry per worker slot). A worker leases one
    for the whole area and returns it when done, so all three stages of an area stay on
    one card and a worker that finishes early picks up the next area on whichever card
    just came free.
    """
    rows = []
    device = slots.get() if slots is not None else None
    area_started = time.time()
    try:
        rows = _run_area_stages(area, args, log_dir, device)
    finally:
        if slots is not None:
            slots.put(device)

    with counter['lock']:
        counter['done'] += 1
        n, total = counter['done'], counter['total']
        rate = (time.time() - counter['start']) / max(n, 1)
        eta = timedelta(seconds=int(rate * (total - n)))
        log(f'[{n}/{total}] {area} finished in '
            f'{timedelta(seconds=int(time.time() - area_started))} | ETA {eta}')
    return rows


def _run_area_stages(area, args, log_dir, device):
    rows = []
    tag = f'{area} gpu{device}' if device is not None else area
    for stage in args.stage_list:
        status, secs, log_path = run_stage(area, stage, args, log_dir, device)
        row = {
            'area_code': area, 'stage': stage, 'status': status,
            'seconds': round(secs, 2), 'duration': str(timedelta(seconds=int(secs))),
            'workers': args.workers, 'gpu': '' if device is None else device,
            'log': os.path.basename(log_path),
            'artifact': '', 'bytes': '',
        }
        rows.append(row)
        if status.startswith('failed') or status in ('timeout',) or status.startswith('error'):
            log(f'  [{tag}] {stage}: {status} -- skipping remaining stages '
                f'(see {os.path.basename(log_path)})')
            break
        if status in ('ok', 'skipped'):
            dest, nbytes = collect_stage(area, stage, args)
            if dest:
                row['artifact'] = os.path.basename(dest)
                row['bytes'] = nbytes
        collected = f' -> {row["artifact"]} ({row["bytes"]} bytes)' if row['artifact'] else ''
        if status == 'ok':
            log(f'  [{tag}] {stage}: ok in {timedelta(seconds=int(secs))}{collected}')
        elif status == 'skipped':
            log(f'  [{tag}] {stage}: already present, skipped{collected}')
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Run individuals -> households -> assignment for many areas.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_argument_group('area selection')
    src.add_argument('--region', choices=['london', 'oxford', 'subset'], default='london',
                     help='built-in area list (default: london, all 983 MSOAs; '
                          'subset = 5-area REPRESENTATIVE_SUBSET from '
                          'greaterLondonAreas.py; oxford = 17 Oxford MSOAs). '
                          'Ignored when --areas or --areas-file is given.')
    src.add_argument('--areas', help='comma-separated area codes')
    src.add_argument('--areas-file', help='file with one area code per line')
    src.add_argument('--limit', type=int, help='process only the first N areas')

    run = ap.add_argument_group('execution')
    run.add_argument('--stages', default='individuals,households,assignment',
                     help='comma-separated subset of individuals,households,assignment')
    run.add_argument('--workers', type=int, default=None,
                     help='areas to run concurrently (default: GPUs found x --per-gpu, '
                          'or 1 on CPU; see timing note)')
    run.add_argument('--gpus', default='auto',
                     help='"auto" to detect CUDA devices (default), "none" to force CPU, '
                          'or an explicit list like 0,1,2')
    run.add_argument('--per-gpu', type=int, default=1,
                     help='worker slots per GPU when --workers is not given (default 1). '
                          'These models are small, so 2-3 raises utilisation.')
    run.add_argument('--threads-per-worker', type=int, default=1,
                     help='OMP/MKL threads per child process (default 1)')
    run.add_argument('--skip-existing', action='store_true',
                     help='skip a stage whose output already exists (resume a sweep)')
    run.add_argument('--timeout', type=int,
                     help='per-stage timeout in seconds')
    run.add_argument('--dry-run', action='store_true',
                     help='print the commands without running them')

    out = ap.add_argument_group('output')
    out.add_argument('--summary', default=None,
                     help='summary CSV path (default outputs/pipeline_summary_<ts>.csv)')
    out.add_argument('--collect-dir', default=None,
                     help='directory for the collected stage artefacts '
                          '(default <repo>/GreaterLondonSyntheticPopulation)')
    out.add_argument('--no-collect', action='store_true',
                     help='do not copy stage tensors into the collected population directory')
    args = ap.parse_args()

    repo_root = os.path.dirname(HERE)
    args.collect_dir = os.path.abspath(
        args.collect_dir or os.path.join(repo_root, 'GreaterLondonSyntheticPopulation'))

    args.stage_list = [s.strip() for s in args.stages.split(',') if s.strip()]
    unknown = [s for s in args.stage_list if s not in STAGE_SCRIPTS]
    if unknown:
        raise SystemExit(f'unknown stage(s): {unknown}. Valid: {list(STAGE_SCRIPTS)}')
    args.stage_list.sort(key=STAGE_ORDER.index)   # enforce dependency order

    gpus = resolve_gpus(args)
    if args.workers is None:
        args.workers = max(1, len(gpus) * max(1, args.per_gpu)) if gpus else 1
    args.workers = max(1, args.workers)
    # One slot per worker, GPUs dealt round-robin across the slots.
    slot_devices = [gpus[i % len(gpus)] for i in range(args.workers)] if gpus else []

    areas = resolve_areas(args)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(HERE, 'outputs', 'logs', stamp)
    os.makedirs(log_dir, exist_ok=True)
    summary_path = args.summary or os.path.join(
        HERE, 'outputs', f'pipeline_summary_{stamp}.csv')

    print('=' * 78)
    print(f'Areas       : {len(areas)}')
    print(f'Stages      : {" -> ".join(args.stage_list)}')
    print(f'Workers     : {args.workers} (threads per worker: {args.threads_per_worker})')
    if gpus:
        counts = {d: slot_devices.count(d) for d in gpus}
        print(f'GPUs        : {len(gpus)} detected -> '
              + ', '.join(f'{d}:{counts[d]} slot(s)' for d in gpus))
        for line in describe_gpus(gpus):
            print(line)
    else:
        print('GPUs        : none in use (CPU)')
    print(f'Logs        : {log_dir}')
    print(f'Summary     : {summary_path}')
    print(f'Population  : {"(collection disabled)" if args.no_collect else args.collect_dir}')
    if args.workers > 1:
        print()
        print('NOTE: with --workers > 1 the per-area durations below include time spent')
        print('      waiting on shared hardware. They are throughput figures, not the')
        print('      cost of synthesising one area. Use --workers 1 for that.')
    print('=' * 78)

    counter = {'done': 0, 'total': len(areas), 'lock': threading.Lock(),
               'start': time.time()}
    all_rows = []

    slots = None
    if slot_devices:
        slots = queue.Queue()
        for dev in slot_devices:
            slots.put(dev)

    if args.workers <= 1:
        for area in areas:
            all_rows.extend(run_area(area, args, log_dir, counter, slots))
    else:
        # Threads, not processes: the real work happens in subprocesses, so threads
        # only need to wait on them, and this avoids Windows pickling constraints.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_area, a, args, log_dir, counter, slots)
                       for a in areas]
            for fut in futures:
                all_rows.extend(fut.result())

    total_elapsed = time.time() - counter['start']

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['area_code', 'stage', 'status', 'seconds',
                                                'duration', 'workers', 'gpu', 'log',
                                                'artifact', 'bytes'])
        writer.writeheader()
        writer.writerows(all_rows)

    print('=' * 78)
    print(f'Total wall-clock: {timedelta(seconds=int(total_elapsed))} '
          f'for {len(areas)} areas at concurrency {args.workers}')
    by_status = {}
    for row in all_rows:
        key = row['status'].split('(')[0]
        by_status[key] = by_status.get(key, 0) + 1
    for status, n in sorted(by_status.items()):
        print(f'  {status:12s} {n}')
    failures = [r for r in all_rows if r['status'].startswith(('failed', 'error', 'timeout'))]
    if failures:
        print('\nFailed stages:')
        for row in failures:
            print(f"  {row['area_code']:12s} {row['stage']:12s} {row['status']:16s} {row['log']}")
        print('\nRe-run just these with --skip-existing to leave completed work alone.')

    if not args.no_collect:
        write_manifest(args.collect_dir, all_rows)

    print(f'\nSummary written to {summary_path}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
