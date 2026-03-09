"""
Main experiment runner for PhysCoT vs Baseline comparison.

Runs:
  Experiment 1: Block toppling — 10 trials × 2 methods = 20 trials
  Experiment 2: Tool selection — 10 trials × 2 methods = 20 trials
  Total: 40 trials

Each trial:
  - resets environment with a varied seed / parameters
  - runs the policy
  - saves a video
  - logs results to JSON
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np

from env.block_toppling import BlockToppingEnv
from env.tool_selection import ToolSelectionEnv
from scripts.policies import BaselinePolicy, PhysCoTPolicy

# ── Output directories ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
VIDEO_BASE  = os.path.join(RESULTS_DIR, 'videos')


# ── Experiment 1: Block toppling ─────────────────────────────────────────────

# Varied trial parameters (block dimensions, friction, position)
BLOCK_TRIALS = [
    # (width_cm, height_cm, mass_kg, friction, seed)
    (7,  20, 0.40, 0.45, 0),
    (8,  22, 0.50, 0.50, 1),
    (9,  18, 0.45, 0.55, 2),
    (6,  24, 0.35, 0.40, 3),
    (10, 16, 0.60, 0.60, 4),
    (7,  21, 0.42, 0.48, 5),
    (8,  19, 0.55, 0.52, 6),
    (9,  23, 0.38, 0.42, 7),
    (6,  20, 0.50, 0.35, 8),
    (11, 17, 0.65, 0.58, 9),
]

# Varied tool-selection trials (object position, obstacle placement)
TOOL_TRIALS = [
    # (obj_x, obj_y, obs_x, obs_y, obs_w, obs_h, goal_x, goal_y, seed)
    (0.45, 0.30, 0.25, 0.22, 0.12, 0.18, 0.10, 0.40, 10),
    (0.50, 0.35, 0.28, 0.25, 0.14, 0.16, 0.08, 0.45, 11),
    (0.40, 0.28, 0.22, 0.20, 0.10, 0.20, 0.12, 0.38, 12),
    (0.48, 0.32, 0.26, 0.28, 0.12, 0.14, 0.09, 0.42, 13),
    (0.42, 0.38, 0.24, 0.30, 0.11, 0.18, 0.11, 0.48, 14),
    (0.55, 0.30, 0.30, 0.22, 0.13, 0.17, 0.07, 0.40, 15),
    (0.45, 0.25, 0.25, 0.18, 0.12, 0.16, 0.10, 0.35, 16),
    (0.50, 0.40, 0.27, 0.32, 0.14, 0.18, 0.08, 0.50, 17),
    (0.38, 0.30, 0.21, 0.24, 0.10, 0.22, 0.13, 0.40, 18),
    (0.52, 0.33, 0.29, 0.26, 0.13, 0.15, 0.08, 0.43, 19),
]


def run_block_trial(policy, trial_params, trial_id, video_dir):
    """Run one block-toppling trial. Returns log dict."""
    w_cm, h_cm, mass, friction, env_seed = trial_params
    w = w_cm / 100.0
    h = h_cm / 100.0

    env = BlockToppingEnv(
        block_width=w, block_height=h, block_mass=mass,
        friction_coef=friction, push_force=5.0,
        dt=0.01, max_steps=400, seed=env_seed
    )

    obs = env.reset()
    scene_desc = env.get_scene_description()
    t0 = time.time()

    contact_frac, reasoning = policy.act_block(obs)
    obs, success, info = env.step(contact_frac, force_direction_sign=1.0)

    elapsed = time.time() - t0

    # Save video
    video_fname = f"trial_{trial_id:02d}.mp4"
    video_path = os.path.join(video_dir, video_fname)
    env.render_video(video_path,
                     title=f"{policy.name.upper()} – Block Toppling Trial {trial_id}")

    # Failure mode classification
    failure_mode = "success"
    if not success:
        if info['max_tilt_deg'] < 10:
            failure_mode = "failed_to_make_contact"
        elif info.get('slid_only', False):
            failure_mode = "pushed_too_low_sliding"
        elif contact_frac < 0.4:
            failure_mode = "pushed_too_low_insufficient_torque"
        elif info['max_tilt_deg'] < 30:
            failure_mode = "insufficient_torque"
        else:
            failure_mode = "unstable_approach"

    log = {
        "experiment_name": "block_toppling",
        "method_name": policy.name,
        "trial_id": trial_id,
        "random_seed": int(env_seed),
        "env_params": {
            "block_width_cm": w_cm,
            "block_height_cm": h_cm,
            "block_mass_kg": mass,
            "friction_coef": friction,
            "aspect_ratio": round(h / w, 2)
        },
        "task_instruction": "Push the upright block over so it topples.",
        "scene_description": scene_desc,
        "reasoning_text": reasoning,
        "action_summary": {
            "contact_height_frac": round(float(contact_frac), 3),
            "contact_height_cm": round(float(contact_frac * h * 100), 1),
            "push_direction": "lateral_right"
        },
        "success": bool(success),
        "metrics": {
            "max_tilt_deg": round(float(info['max_tilt_deg']), 2),
            "final_theta_deg": round(float(info['final_theta_deg']), 2),
            "slid_only": bool(info.get('slid_only', False)),
            "steps": int(info['steps'])
        },
        "failure_mode": failure_mode,
        "video_path": os.path.relpath(video_path, BASE_DIR),
        "elapsed_s": round(elapsed, 3)
    }
    return log


def run_tool_trial(policy, trial_params, trial_id, video_dir):
    """Run one tool-selection trial. Returns log dict."""
    obj_x, obj_y, obs_x, obs_y, obs_w, obs_h, goal_x, goal_y, env_seed = trial_params

    env = ToolSelectionEnv(
        obj_x=obj_x, obj_y=obj_y,
        obs_x=obs_x, obs_y=obs_y,
        obs_w=obs_w, obs_h=obs_h,
        goal_x=goal_x, goal_y=goal_y,
        hook_x=0.05, hook_y=0.10,
        str_x=0.05, str_y=0.25,
        obj_radius=0.04, goal_radius=0.08,
        seed=env_seed
    )

    obs = env.reset()
    scene_desc = env.get_scene_description()
    t0 = time.time()

    tool_chosen, reasoning = policy.act_tool(obs)
    obs, success, info = env.step(tool_chosen)

    elapsed = time.time() - t0

    # Save video
    video_fname = f"trial_{trial_id:02d}.mp4"
    video_path = os.path.join(video_dir, video_fname)
    env.render_video(video_path,
                     title=f"{policy.name.upper()} – Tool Selection Trial {trial_id}")

    # Failure mode
    failure_mode = "success"
    if not success:
        if tool_chosen != info['correct_tool']:
            if not info['path_feasible']:
                failure_mode = "wrong_tool_path_blocked"
            else:
                failure_mode = "wrong_tool_inefficient"
        elif not info['path_feasible']:
            failure_mode = "correct_tool_wrong_grasp"
        else:
            failure_mode = "moved_wrong_direction"

    log = {
        "experiment_name": "tool_selection",
        "method_name": policy.name,
        "trial_id": trial_id,
        "random_seed": int(env_seed),
        "env_params": {
            "obj_x": obj_x, "obj_y": obj_y,
            "obs_x": obs_x, "obs_y": obs_y,
            "obs_w": obs_w, "obs_h": obs_h,
            "goal_x": goal_x, "goal_y": goal_y
        },
        "task_instruction": "Select the correct tool and move the object to the goal.",
        "scene_description": scene_desc,
        "reasoning_text": reasoning,
        "action_summary": {
            "tool_chosen": tool_chosen,
            "correct_tool": info['correct_tool'],
            "tool_correct": tool_chosen == info['correct_tool']
        },
        "success": bool(success),
        "metrics": {
            "tool_correct": bool(tool_chosen == info['correct_tool']),
            "path_feasible": bool(info['path_feasible']),
            "dist_to_goal_cm": round(float(info['dist_to_goal']) * 100, 2),
            "direct_blocked": bool(info['direct_blocked'])
        },
        "failure_mode": failure_mode,
        "video_path": os.path.relpath(video_path, BASE_DIR),
        "elapsed_s": round(elapsed, 3)
    }
    return log


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    policies = [BaselinePolicy(seed=100), PhysCoTPolicy(seed=200)]
    all_logs = []

    # ── Experiment 1: Block Toppling ──────────────────────────────────────────
    print("=" * 60)
    print("EXPERIMENT 1: Block Toppling")
    print("=" * 60)

    for policy in policies:
        vid_dir = os.path.join(VIDEO_BASE, 'exp1_block_toppling', policy.name)
        os.makedirs(vid_dir, exist_ok=True)
        print(f"\nRunning {policy.name.upper()} policy ({len(BLOCK_TRIALS)} trials)...")

        for trial_id, params in enumerate(BLOCK_TRIALS):
            print(f"  Trial {trial_id + 1:02d}/{len(BLOCK_TRIALS)}...", end=' ', flush=True)
            log = run_block_trial(policy, params, trial_id, vid_dir)
            all_logs.append(log)
            status = "SUCCESS ✓" if log['success'] else f"FAIL ({log['failure_mode']})"
            print(f"{status}  [tilt={log['metrics']['max_tilt_deg']:.1f}°, "
                  f"contact={log['action_summary']['contact_height_frac']:.2f}]")

        successes = sum(l['success'] for l in all_logs
                        if l['experiment_name'] == 'block_toppling'
                        and l['method_name'] == policy.name)
        print(f"  → {policy.name.upper()} success rate: {successes}/{len(BLOCK_TRIALS)}")

    # ── Experiment 2: Tool Selection ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Tool Selection")
    print("=" * 60)

    for policy in policies:
        vid_dir = os.path.join(VIDEO_BASE, 'exp2_tool_selection', policy.name)
        os.makedirs(vid_dir, exist_ok=True)
        print(f"\nRunning {policy.name.upper()} policy ({len(TOOL_TRIALS)} trials)...")

        for trial_id, params in enumerate(TOOL_TRIALS):
            print(f"  Trial {trial_id + 1:02d}/{len(TOOL_TRIALS)}...", end=' ', flush=True)
            log = run_tool_trial(policy, params, trial_id, vid_dir)
            all_logs.append(log)
            tc = log['action_summary']['tool_correct']
            status = "SUCCESS ✓" if log['success'] else f"FAIL ({log['failure_mode']})"
            print(f"{status}  [tool={log['action_summary']['tool_chosen']}, "
                  f"correct={log['action_summary']['correct_tool']}]")

        successes = sum(l['success'] for l in all_logs
                        if l['experiment_name'] == 'tool_selection'
                        and l['method_name'] == policy.name)
        print(f"  → {policy.name.upper()} success rate: {successes}/{len(TOOL_TRIALS)}")

    # ── Save all logs ─────────────────────────────────────────────────────────
    os.makedirs(METRICS_DIR, exist_ok=True)
    all_logs_path = os.path.join(METRICS_DIR, 'all_trials.json')
    with open(all_logs_path, 'w') as f:
        json.dump(all_logs, f, indent=2)
    print(f"\nAll logs saved to {all_logs_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp in ['block_toppling', 'tool_selection']:
        for method in ['baseline', 'physcot']:
            trials = [l for l in all_logs
                      if l['experiment_name'] == exp and l['method_name'] == method]
            n_success = sum(l['success'] for l in trials)
            rate = n_success / len(trials) * 100 if trials else 0
            print(f"  {exp:20s}  {method:10s}  {n_success}/{len(trials)}  ({rate:.0f}%)")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    import csv
    summary_path = os.path.join(METRICS_DIR, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'method', 'trial_id', 'success', 'failure_mode'])
        for l in all_logs:
            writer.writerow([l['experiment_name'], l['method_name'],
                             l['trial_id'], int(l['success']), l['failure_mode']])
    print(f"Summary CSV saved to {summary_path}")


if __name__ == '__main__':
    main()
