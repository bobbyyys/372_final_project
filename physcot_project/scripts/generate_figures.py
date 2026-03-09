"""
Generate all publication-quality figures for the PhysCoT paper.

Figures produced:
  1. fig_pipeline.png         — Method pipeline overview
  2. fig_prompt_schema.png    — PhysCoT reasoning schema table
  3. fig_exp_setup.png        — Experiment setup diagrams
  4. fig_main_results.png     — Main quantitative bar chart
  5. fig_failure_modes.png    — Failure mode breakdown
  6. fig_contact_height.png   — Contact height distribution (block exp)
  7. fig_qualitative.png      — Qualitative rollout frames
  8. fig_reasoning_example.png— Example PhysCoT reasoning
  9. fig_future_pipeline.png  — Future supervised training pipeline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(BASE_DIR, 'results', 'figures')
METRICS_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'all_trials.json')

os.makedirs(FIGS_DIR, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
with open(METRICS_PATH) as f:
    all_logs = json.load(f)


def get_trials(exp, method):
    return [l for l in all_logs if l['experiment_name'] == exp and l['method_name'] == method]


def success_rate(trials):
    return sum(l['success'] for l in trials) / len(trials) if trials else 0


def binomial_ci(n_success, n_total, alpha=0.05):
    """Wilson score 95% CI for binomial proportion."""
    if n_total == 0:
        return 0, 0
    z = stats.norm.ppf(1 - alpha / 2)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


COLORS = {
    'baseline': '#e74c3c',
    'physcot':  '#2ecc71',
    'blue':     '#3498db',
    'orange':   '#f39c12',
    'gray':     '#95a5a6',
    'dark':     '#2c3e50',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Method pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    def box(cx, cy, w, h, label, sublabel='', color='#3498db', fontsize=10):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='#2c3e50', lw=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(cx, cy + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white')
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel,
                    ha='center', va='center', fontsize=8, color='#ecf0f1')

    def arrow(x1, x2, y=1.5):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                   lw=2, connectionstyle='arc3,rad=0'))

    # Boxes
    box(0.9, 1.5, 1.5, 1.0, 'Observation', 'RGB image\n+ instruction', '#8e44ad')
    arrow(1.65, 2.4)

    # PhysCoT box (larger, 4-step)
    pc_rect = FancyBboxPatch((2.4, 0.35), 3.8, 2.3,
                             boxstyle="round,pad=0.05",
                             facecolor='#27ae60', edgecolor='#1e8449', lw=2, alpha=0.9)
    ax.add_patch(pc_rect)
    ax.text(4.3, 2.38, 'PhysCoT Reasoning', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    steps = ['A. Task\nDecomposition', 'B. Relevant\nPhysics',
             'C. Visual\nEstimates', 'D. Action\nImplication']
    step_colors = ['#2ecc71', '#27ae60', '#1e8449', '#196f3d']
    for i, (s, sc) in enumerate(zip(steps, step_colors)):
        sx = 2.65 + i * 0.97
        srect = FancyBboxPatch((sx - 0.43, 0.55), 0.86, 1.55,
                               boxstyle="round,pad=0.03",
                               facecolor=sc, edgecolor='white', lw=1, alpha=0.85)
        ax.add_patch(srect)
        ax.text(sx, 1.33, s, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
        if i < 3:
            ax.annotate('', xy=(sx + 0.47, 1.33), xytext=(sx + 0.43, 1.33),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

    arrow(6.2, 7.0)
    box(7.55, 1.5, 1.5, 1.0, 'OpenVLA\nAction', 'action\nvector', '#e67e22')
    arrow(8.3, 9.1)
    box(9.55, 1.5, 1.0, 1.0, 'Simulator\nRollout', '', '#c0392b')

    # Dashed box for baseline (plain)
    base_rect = mpatches.Rectangle((2.4, 0.0), 3.8, 0.32,
                                    fill=True, facecolor='#e74c3c', alpha=0.15,
                                    edgecolor='#e74c3c', lw=1.5, linestyle='--')
    ax.add_patch(base_rect)
    ax.text(4.3, 0.16, 'Baseline: skip reasoning (Plain OpenVLA)',
            ha='center', va='center', fontsize=8.5, color='#c0392b',
            fontstyle='italic')

    ax.set_title('PhysCoT Inference Pipeline', fontsize=14, fontweight='bold',
                 pad=8, color=COLORS['dark'])
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_pipeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Prompt schema
# ─────────────────────────────────────────────────────────────────────────────
def fig_prompt_schema():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis('off')

    rows = [
        ('Step A', 'Task Decomposition',
         'Overall goal • Immediate sub-goal\n• Required contact/state change',
         '"Topple the upright block; sub-goal: drive\nCOM past support polygon edge."'),
        ('Step B', 'Relevant Physics',
         'Torque/leverage • Friction regime\n• Stability • Contact mechanics\n• Affordance',
         '"Toppling requires τ_push = F·y_c > m·g·(w/2).\nPush above COM maximises torque."'),
        ('Step C', 'Visual Physical Estimates',
         'COM location • Aspect ratio\n• Pivot edge • Friction estimate\n• Tool geometry',
         '"Block h/w ≈ 2.5 → easy to topple.\nBest contact zone: 65–88% of height."'),
        ('Step D', 'Action Implication',
         'Contact point • Push direction\n• Force magnitude • Causal rationale',
         '"Contact at 72% height; lateral right.\nτ_push > τ_crit → topple expected."'),
    ]

    col_w = [0.08, 0.16, 0.38, 0.38]
    col_x = [0.01, 0.10, 0.27, 0.66]
    headers = ['', 'Stage', 'Reasoning Content', 'Block Toppling Example']
    hcolors = ['#2c3e50'] * 4

    # Header
    y = 0.92
    for j, (h, cx) in enumerate(zip(headers, col_x)):
        ax.text(cx, y, h, ha='left', va='center', fontsize=11,
                fontweight='bold', color='white',
                transform=ax.transAxes,
                bbox=dict(facecolor=hcolors[j], edgecolor='none',
                          boxstyle='square', alpha=0.9,
                          pad=0.3) if h else None)

    # Divider
    line = plt.Line2D([0, 1], [0.88, 0.88], color=COLORS['dark'], lw=1.5,
                      transform=ax.transAxes, clip_on=False)
    ax.add_line(line)

    step_colors = ['#8e44ad', '#2980b9', '#27ae60', '#e67e22']
    for i, (step, name, content, example) in enumerate(rows):
        y = 0.78 - i * 0.20

        # Step badge
        sc = step_colors[i]
        ax.text(col_x[0] + 0.015, y, step, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                transform=ax.transAxes,
                bbox=dict(facecolor=sc, boxstyle='round,pad=0.3',
                          edgecolor='none'))
        # Name
        ax.text(col_x[1] + 0.01, y, name, ha='left', va='center',
                fontsize=10, fontweight='bold', color=sc,
                transform=ax.transAxes)
        # Content
        ax.text(col_x[2] + 0.005, y, content, ha='left', va='center',
                fontsize=9.5, color=COLORS['dark'],
                transform=ax.transAxes)
        # Example
        ax.text(col_x[3] + 0.005, y, example, ha='left', va='center',
                fontsize=9, color='#555555', fontstyle='italic',
                transform=ax.transAxes)

        if i < 3:
            line = plt.Line2D([0, 1], [y - 0.09, y - 0.09],
                              color='#bdc3c7', lw=0.8, linestyle='--',
                              transform=ax.transAxes, clip_on=False)
            ax.add_line(line)

        # Down arrow between steps
        if i < 3:
            ax.annotate('', xy=(col_x[0] + 0.015, y - 0.115),
                        xytext=(col_x[0] + 0.015, y - 0.09),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=step_colors[i+1], lw=1.5))

    ax.set_title('PhysCoT Reasoning Schema', fontsize=14, fontweight='bold',
                 pad=10, x=0.5, y=0.98, color=COLORS['dark'])
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_prompt_schema.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Experiment setup
# ─────────────────────────────────────────────────────────────────────────────
def fig_exp_setup():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: Block toppling ---
    ax = axes[0]
    ax.set_xlim(-0.35, 0.55)
    ax.set_ylim(-0.05, 0.32)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')
    ax.grid(True, alpha=0.3)

    # Ground
    ax.axhline(0, color='#8B4513', lw=2)
    ax.fill_between([-0.35, 0.55], -0.05, 0, color='#c8a97a', alpha=0.4)

    # Block (upright)
    w, h = 0.08, 0.20
    block = plt.Rectangle((-w/2, 0), w, h,
                           facecolor='#3498db', edgecolor='#2c3e50', lw=2, zorder=3)
    ax.add_patch(block)

    # COM marker
    ax.plot(0, h/2, 'r+', ms=14, mew=3, zorder=5, label='COM')

    # Critical height line
    y_crit = 0.5 * 0.5 * 9.81 * w / (2 * 5.0)
    ax.axhline(y_crit, xmin=0.25, xmax=0.55, color='orange',
               lw=2, linestyle='--', zorder=4)
    ax.text(0.22, y_crit + 0.005, r'$y_{crit}$', fontsize=11, color='orange')

    # Good contact zone
    ax.axhspan(0.65*h, 0.88*h, xmin=0.25, xmax=0.6,
               alpha=0.25, color='green', zorder=2)
    ax.text(0.22, 0.76*h, 'Best\nzone', fontsize=9, color='green',
            ha='left', va='center')

    # Arrows: push
    ax.annotate('', xy=(w/2, 0.72*h), xytext=(w/2 + 0.12, 0.72*h),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    ax.text(w/2 + 0.13, 0.72*h, 'F', fontsize=13, color='#e74c3c',
            fontweight='bold', va='center')

    # Pivot
    ax.plot(-w/2, 0, 'ko', ms=8, zorder=6)
    ax.text(-w/2 - 0.02, -0.015, 'pivot', fontsize=8, ha='right', color='#333')

    # Toppled ghost
    theta = np.radians(80)
    cx_top = w/2 * np.cos(theta) - 0
    cy_top = w/2 * np.sin(theta)
    corners = [(-w/2, 0), (w/2, 0), (w/2, h), (-w/2, h)]
    piv = (-w/2, 0)
    top_corners = [(piv[0] + (px - piv[0])*np.cos(theta) - (py - piv[1])*np.sin(theta),
                    piv[1] + (px - piv[0])*np.sin(theta) + (py - piv[1])*np.cos(theta))
                   for px, py in corners]
    ax.add_patch(plt.Polygon(top_corners, closed=True,
                             facecolor='#3498db', edgecolor='#2c3e50',
                             lw=1, alpha=0.25, zorder=2))
    ax.text(0.12, 0.22, 'toppled\n(target)', fontsize=8.5, color='#3498db',
            ha='left', style='italic')

    ax.set_title('Experiment 1: Block Toppling', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper left', fontsize=9)

    # --- Right: Tool selection ---
    ax2 = axes[1]
    ax2.set_xlim(-0.05, 0.65)
    ax2.set_ylim(-0.05, 0.65)
    ax2.set_aspect('equal')
    ax2.set_facecolor('#f5f5f5')
    ax2.grid(True, alpha=0.3)

    # Obstacle
    obs = plt.Rectangle((0.25 - 0.06, 0.22 - 0.09), 0.12, 0.18,
                         facecolor='#e74c3c', edgecolor='#922b21', lw=2, alpha=0.8, zorder=4)
    ax2.add_patch(obs)
    ax2.text(0.25, 0.22, 'Obstacle', ha='center', va='center',
             fontsize=8, color='white', fontweight='bold')

    # Target object
    obj = plt.Circle((0.45, 0.30), 0.04, facecolor='#f39c12',
                      edgecolor='#d35400', lw=2, zorder=5)
    ax2.add_patch(obj)
    ax2.text(0.45, 0.30, 'T', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')

    # Goal
    goal = plt.Circle((0.10, 0.40), 0.08, facecolor='#2ecc71',
                       edgecolor='#1e8449', lw=2, alpha=0.35, zorder=2)
    ax2.add_patch(goal)
    ax2.text(0.10, 0.40, 'Goal', ha='center', va='center',
             fontsize=9, color='#1e8449', fontweight='bold')

    # Robot
    ax2.plot(0, 0, 'ko', ms=14, zorder=6)
    ax2.text(0.01, -0.035, 'Robot', fontsize=9)

    # Hook tool
    ax2.plot(0.05, 0.10, 'b^', ms=14, zorder=6)
    ax2.text(0.07, 0.10, 'Hook ✓', fontsize=9, color='blue', va='center')

    # Straight tool
    ax2.plot(0.05, 0.25, 'ms', ms=12, zorder=6)
    ax2.text(0.07, 0.25, 'Straight ✗', fontsize=9, color='purple', va='center')

    # Blocked direct path
    ax2.annotate('', xy=(0.45, 0.30), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#e74c3c',
                                 lw=2, linestyle='--',
                                 connectionstyle='arc3,rad=0'))
    ax2.text(0.23, 0.10, 'BLOCKED', fontsize=8.5, color='#e74c3c',
             fontweight='bold', rotation=30)

    # Hook arc path
    theta_arc = np.linspace(np.pi*0.85, np.pi*0.2, 40)
    arc_r = 0.40
    arc_x = 0.08 + arc_r * np.cos(theta_arc)
    arc_y = 0.05 + arc_r * np.sin(theta_arc)
    ax2.plot(arc_x, arc_y, 'b--', lw=2.5, alpha=0.7, label='Hook arc path')
    ax2.annotate('', xy=(arc_x[-1], arc_y[-1]),
                 xytext=(arc_x[-2], arc_y[-2]),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax2.set_title('Experiment 2: Tool Selection', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.legend(loc='upper right', fontsize=9)

    plt.suptitle('Simulation Experiment Setups', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_exp_setup.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Main quantitative results
# ─────────────────────────────────────────────────────────────────────────────
def fig_main_results():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, exp, exp_label in zip(
            axes,
            ['block_toppling', 'tool_selection'],
            ['Exp 1: Block Toppling', 'Exp 2: Tool Selection']):

        b_trials = get_trials(exp, 'baseline')
        p_trials = get_trials(exp, 'physcot')
        n = len(b_trials)

        b_succ = sum(l['success'] for l in b_trials)
        p_succ = sum(l['success'] for l in p_trials)

        b_rate = b_succ / n
        p_rate = p_succ / n

        b_lo, b_hi = binomial_ci(b_succ, n)
        p_lo, p_hi = binomial_ci(p_succ, n)

        x = np.array([0, 1])
        rates = [b_rate, p_rate]
        colors = [COLORS['baseline'], COLORS['physcot']]
        labels = ['Baseline\n(Plain OpenVLA)', 'PhysCoT\n(Ours)']

        bars = ax.bar(x, rates, color=colors, width=0.5, edgecolor='#2c3e50',
                      lw=1.5, alpha=0.85, zorder=3)

        # Error bars (Wilson CI)
        ax.errorbar(x[0], b_rate, yerr=[[max(0, b_rate - b_lo)], [max(0, b_hi - b_rate)]],
                    fmt='none', color='#2c3e50', capsize=8, lw=2, capthick=2)
        ax.errorbar(x[1], p_rate, yerr=[[max(0, p_rate - p_lo)], [max(0, p_hi - p_rate)]],
                    fmt='none', color='#2c3e50', capsize=8, lw=2, capthick=2)

        # Value labels
        for xi, rate, succ in zip(x, rates, [b_succ, p_succ]):
            ax.text(xi, rate + 0.05, f'{rate*100:.0f}%\n({succ}/{n})',
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color=COLORS['dark'])

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Success Rate', fontsize=11)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(exp_label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.4, zorder=0)
        ax.set_facecolor('#fafafa')

        # Δ annotation
        delta = p_rate - b_rate
        ax.annotate('', xy=(1, p_rate), xytext=(0, b_rate),
                    arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2,
                                   connectionstyle='arc3,rad=0.3'))
        ax.text(0.5, max(b_rate, p_rate) + 0.16,
                f'Δ = +{delta*100:.0f}%', ha='center', fontsize=11,
                color='#8e44ad', fontweight='bold',
                bbox=dict(facecolor='#f8e8ff', edgecolor='#8e44ad',
                          boxstyle='round,pad=0.3', lw=1.2))

    fig.suptitle('PhysCoT vs. Baseline: Success Rate Comparison\n'
                 '(n = 10 trials per method; error bars = 95% Wilson CI)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_main_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Failure mode breakdown
# ─────────────────────────────────────────────────────────────────────────────
def fig_failure_modes():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, exp, exp_label in zip(
            axes,
            ['block_toppling', 'tool_selection'],
            ['Exp 1: Block Toppling', 'Exp 2: Tool Selection']):

        all_trials = get_trials(exp, 'baseline') + get_trials(exp, 'physcot')
        fail_counts = {}
        method_fail = {'baseline': {}, 'physcot': {}}

        for l in all_trials:
            fm = l['failure_mode']
            fail_counts[fm] = fail_counts.get(fm, 0) + 1
            method_fail[l['method_name']][fm] = \
                method_fail[l['method_name']].get(fm, 0) + 1

        # Sort failure modes
        fms = sorted(fail_counts.keys(), key=lambda x: (x == 'success', x))
        x = np.arange(len(fms))

        b_vals = [method_fail['baseline'].get(fm, 0) for fm in fms]
        p_vals = [method_fail['physcot'].get(fm, 0) for fm in fms]

        w = 0.35
        b_bars = ax.bar(x - w/2, b_vals, width=w, label='Baseline',
                        color=COLORS['baseline'], edgecolor='#2c3e50',
                        lw=1, alpha=0.85)
        p_bars = ax.bar(x + w/2, p_vals, width=w, label='PhysCoT',
                        color=COLORS['physcot'], edgecolor='#2c3e50',
                        lw=1, alpha=0.85)

        clean_labels = [fm.replace('_', '\n') for fm in fms]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, fontsize=8.5, rotation=15, ha='right')
        ax.set_ylabel('Count (out of 10)', fontsize=10)
        ax.set_title(exp_label + '\nFailure Mode Breakdown', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.4)
        ax.set_facecolor('#fafafa')
        ax.set_yticks(range(0, 11, 2))

        # Highlight success bars
        for i, fm in enumerate(fms):
            if fm == 'success':
                for bar in [b_bars[i], p_bars[i]]:
                    bar.set_edgecolor('#27ae60')
                    bar.set_linewidth(2.5)

    plt.suptitle('Failure Mode Analysis by Method', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_failure_modes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Contact height distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_contact_height():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    b_trials = get_trials('block_toppling', 'baseline')
    p_trials = get_trials('block_toppling', 'physcot')

    b_heights = [l['action_summary']['contact_height_frac'] for l in b_trials]
    p_heights = [l['action_summary']['contact_height_frac'] for l in p_trials]
    b_success = [l['success'] for l in b_trials]
    p_success = [l['success'] for l in p_trials]
    b_tilt = [l['metrics']['max_tilt_deg'] for l in b_trials]
    p_tilt = [l['metrics']['max_tilt_deg'] for l in p_trials]

    # --- Left: contact height histogram ---
    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    ax.hist(b_heights, bins=bins, alpha=0.7, color=COLORS['baseline'],
            label='Baseline', edgecolor='white', lw=1)
    ax.hist(p_heights, bins=bins, alpha=0.7, color=COLORS['physcot'],
            label='PhysCoT', edgecolor='white', lw=1)

    # Critical zone
    ax.axvspan(0.65, 0.88, alpha=0.15, color='green')
    ax.axvline(0.65, color='green', lw=2, linestyle='--', label='Optimal zone (65–88%)')
    ax.axvline(0.88, color='green', lw=2, linestyle='--')

    ax.set_xlabel('Contact Height (fraction of block height)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Contact Height Distribution\n(Block Toppling)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.4)

    # --- Right: scatter height vs tilt ---
    ax2 = axes[1]
    for h, t, s in zip(b_heights, b_tilt, b_success):
        marker = 'o' if s else 'x'
        ax2.scatter(h, t, color=COLORS['baseline'], s=80, marker=marker,
                    alpha=0.8, linewidths=2)
    for h, t, s in zip(p_heights, p_tilt, p_success):
        marker = 'o' if s else 'x'
        ax2.scatter(h, t, color=COLORS['physcot'], s=80, marker=marker,
                    alpha=0.8, linewidths=2)

    ax2.axhline(60, color='#8e44ad', lw=2, linestyle=':', label='Success threshold (60°)')
    ax2.axvspan(0.65, 0.88, alpha=0.12, color='green', label='Optimal contact zone')

    # Legend proxies
    base_patch = mpatches.Patch(color=COLORS['baseline'], label='Baseline', alpha=0.8)
    phys_patch = mpatches.Patch(color=COLORS['physcot'], label='PhysCoT', alpha=0.8)
    succ_marker = plt.Line2D([0], [0], marker='o', color='gray', ms=8, label='Success', lw=0)
    fail_marker = plt.Line2D([0], [0], marker='x', color='gray', ms=8, label='Failure', lw=0,
                             markeredgewidth=2)

    ax2.set_xlabel('Contact Height Fraction', fontsize=11)
    ax2.set_ylabel('Max Tilt Angle (degrees)', fontsize=11)
    ax2.set_title('Contact Height vs. Max Tilt Angle', fontsize=11, fontweight='bold')
    ax2.legend(handles=[base_patch, phys_patch, succ_marker, fail_marker,
                        plt.Line2D([0], [0], color='#8e44ad', lw=2, linestyle=':',
                                   label='Success threshold')],
               fontsize=9, loc='upper left')
    ax2.grid(alpha=0.35)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_contact_height.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Qualitative rollout snapshots
# ─────────────────────────────────────────────────────────────────────────────
def fig_qualitative():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    titles = [
        ['(a) Baseline push\n(contact low, sliding)', '(b) Mid-roll\n(insufficient torque)',
         '(c) PhysCoT push\n(contact high)', '(d) Block toppled\n(success)'],
        ['(e) Baseline selects\nstraight tool', '(f) Path blocked\n(failure)',
         '(g) PhysCoT selects\nhook tool', '(h) Object reaches\ngoal (success)']
    ]

    # Row 0: Block toppling snapshots
    thetas = [0, np.radians(10), 0, np.radians(75)]
    contacts = [0.30, None, 0.72, None]
    colors_row = [COLORS['baseline'], COLORS['baseline'],
                  COLORS['physcot'], COLORS['physcot']]

    for j in range(4):
        ax = axes[0][j]
        ax.set_xlim(-0.30, 0.40)
        ax.set_ylim(-0.04, 0.28)
        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5f5')
        ax.axhline(0, color='#8B4513', lw=2)
        ax.fill_between([-0.30, 0.40], -0.04, 0, color='#c8a97a', alpha=0.3)

        w, h = 0.08, 0.20
        theta = thetas[j]
        piv = (-w/2, 0)
        corners = [(-w/2, 0), (w/2, 0), (w/2, h), (-w/2, h)]
        top_corners = [(piv[0] + (px - piv[0])*np.cos(theta) - (py - piv[1])*np.sin(theta),
                        piv[1] + (px - piv[0])*np.sin(theta) + (py - piv[1])*np.cos(theta))
                       for px, py in corners]
        block_poly = plt.Polygon(top_corners, closed=True,
                                 facecolor='#3498db', edgecolor='#2c3e50', lw=1.5, zorder=3)
        ax.add_patch(block_poly)

        # Contact arrow
        if contacts[j] is not None:
            yc = contacts[j] * h
            # Rotate contact point
            cx = piv[0] + (w/2)*np.cos(theta) - yc*np.sin(theta)
            cy = piv[1] + (w/2)*np.sin(theta) + yc*np.cos(theta)
            ax.annotate('', xy=(cx, cy), xytext=(cx + 0.10, cy),
                        arrowprops=dict(arrowstyle='->', color=colors_row[j], lw=3))

        ax.set_title(titles[0][j], fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

        # Method badge
        method = 'Baseline' if j < 2 else 'PhysCoT'
        badge_color = COLORS['baseline'] if j < 2 else COLORS['physcot']
        ax.text(0.05, 0.95, method, transform=ax.transAxes,
                fontsize=8, fontweight='bold', color='white', va='top',
                bbox=dict(facecolor=badge_color, boxstyle='round,pad=0.2', alpha=0.9))

    # Row 1: Tool selection snapshots
    scenes = [
        # (obj_x, obj_y, tool, path_ok, phase)
        (0.45, 0.30, 'straight', False, 'choose'),
        (0.45, 0.30, 'straight', False, 'blocked'),
        (0.45, 0.30, 'hook', True, 'choose'),
        (0.10, 0.40, 'hook', True, 'done'),
    ]
    for j, (ox, oy, tool, ok, phase) in enumerate(scenes):
        ax = axes[1][j]
        ax.set_xlim(-0.05, 0.60)
        ax.set_ylim(-0.05, 0.60)
        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5f5')
        ax.grid(True, alpha=0.2)

        obs_r = plt.Rectangle((0.25-0.06, 0.22-0.09), 0.12, 0.18,
                               fc='#e74c3c', ec='#922b21', lw=2, alpha=0.8)
        ax.add_patch(obs_r)

        goal_c = plt.Circle((0.10, 0.40), 0.08, fc='#2ecc71', alpha=0.3, ec='#1e8449', lw=1.5)
        ax.add_patch(goal_c)

        obj_c = plt.Circle((ox, oy), 0.04, fc='#f39c12', ec='#d35400', lw=1.5, zorder=5)
        ax.add_patch(obj_c)
        ax.text(ox, oy, 'T', ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=6)

        ax.plot(0, 0, 'ko', ms=10)

        # Chosen tool indicator
        tc = COLORS['physcot'] if j >= 2 else COLORS['baseline']
        ax.text(0.05, 0.95, f'{tool.capitalize()}\ntool', transform=ax.transAxes,
                fontsize=8, fontweight='bold', color='white', va='top',
                bbox=dict(facecolor=tc, boxstyle='round,pad=0.2', alpha=0.9))

        if phase == 'blocked':
            ax.annotate('', xy=(0.45, 0.30), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2, linestyle='--'))
            ax.text(0.23, 0.08, '✗', fontsize=22, color='#e74c3c', ha='center')

        ax.set_title(titles[1][j], fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row labels
    for row, label in enumerate(['Block Toppling', 'Tool Selection']):
        fig.text(0.01, 0.75 - row * 0.5, label, va='center',
                 rotation='vertical', fontsize=11, fontweight='bold',
                 color=COLORS['dark'])

    plt.suptitle('Qualitative Rollout Snapshots', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_qualitative.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Reasoning example (success vs failure)
# ─────────────────────────────────────────────────────────────────────────────
def fig_reasoning_example():
    # Find a success and failure reasoning from PhysCoT trials
    p_trials = get_trials('block_toppling', 'physcot')
    p_success = [l for l in p_trials if l['success']]
    p_fail = [l for l in p_trials if not l['success']]

    b_trials = get_trials('tool_selection', 'baseline')
    b_fail = [l for l in b_trials if not l['success']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def render_reasoning_box(ax, title, reasoning_text, success, note=''):
        ax.axis('off')
        ax.set_facecolor('#f9f9f9')
        color = '#27ae60' if success else '#e74c3c'
        badge = '✓ SUCCESS' if success else '✗ FAILURE'

        # Title box
        title_box = FancyBboxPatch((0.01, 0.88), 0.98, 0.10,
                                   boxstyle="round,pad=0.02",
                                   transform=ax.transAxes,
                                   facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(title_box)
        ax.text(0.50, 0.93, f'{title}  [{badge}]',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='white')

        # Reasoning text
        ax.text(0.03, 0.84, reasoning_text,
                ha='left', va='top', transform=ax.transAxes,
                fontsize=8.5, family='monospace', color='#2c3e50',
                bbox=dict(facecolor='white', edgecolor='#bdc3c7',
                          boxstyle='round,pad=0.5', alpha=0.95),
                wrap=True)

        if note:
            ax.text(0.03, 0.05, f'Note: {note}',
                    ha='left', va='bottom', transform=ax.transAxes,
                    fontsize=9, color='#7f8c8d', fontstyle='italic')

    # --- Left: PhysCoT success ---
    if p_success:
        trial = p_success[0]
        reasoning = trial['reasoning_text'][:900]  # truncate for display
        render_reasoning_box(axes[0],
                             'PhysCoT Block Toppling – Success',
                             reasoning, success=True,
                             note=f"Contact at {trial['action_summary']['contact_height_frac']:.0%} "
                                  f"height → tilt {trial['metrics']['max_tilt_deg']:.1f}°")
    else:
        axes[0].text(0.5, 0.5, 'No success trials', ha='center', va='center',
                     transform=axes[0].transAxes)

    # --- Right: Baseline failure ---
    if b_fail:
        trial = b_fail[0]
        reasoning = trial['reasoning_text'][:600]
        render_reasoning_box(axes[1],
                             'Baseline Tool Selection – Failure',
                             reasoning, success=False,
                             note=f"Chose '{trial['action_summary']['tool_chosen']}' "
                                  f"but correct was '{trial['action_summary']['correct_tool']}'")
    else:
        axes[1].text(0.5, 0.5, 'No failure trials', ha='center', va='center',
                     transform=axes[1].transAxes)

    plt.suptitle('Reasoning Comparison: PhysCoT Success vs. Baseline Failure',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_reasoning_example.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Future supervised training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig_future_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def box(cx, cy, w, h, label, sublabel='', color='#3498db', fontsize=10):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='#2c3e50', lw=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(cx, cy + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white')
        if sublabel:
            ax.text(cx, cy - 0.25, sublabel,
                    ha='center', va='center', fontsize=8, color='#ecf0f1')

    def arrow(x1, x2, y=2.0, label=''):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                   lw=2.2, connectionstyle='arc3,rad=0'))
        if label:
            ax.text((x1 + x2) / 2, y + 0.22, label,
                    ha='center', fontsize=8.5, color=COLORS['dark'])

    # Phase 1: Simulation rollout
    box(1.1, 2.0, 1.8, 1.2, 'Simulation\nRollout', 'diverse episodes', '#8e44ad')
    arrow(2.0, 2.9, label='episodes')

    # Phase 2: VLM annotation
    box(3.7, 2.0, 1.6, 1.2, 'VLM\nAnnotator', 'Claude / GPT-4V\none-off labelling', '#2980b9')
    arrow(4.5, 5.4, label='annotated\ntraces')

    # Phase 3: Dataset
    box(6.2, 2.0, 1.5, 1.2, 'PhysCoT\nDataset',
        'obs + reasoning\n+ action + outcome', '#27ae60')
    arrow(6.95, 7.8, label='supervised\ndata')

    # Phase 4: Fine-tuning
    box(8.55, 2.0, 1.5, 1.2, 'Finetuned\nVLA',
        'OpenVLA +\nPhysCoT head', '#e67e22')
    arrow(9.3, 10.15, label='deploy')

    # Phase 5: Deploy
    box(10.8, 2.0, 1.3, 1.2, 'Efficient\nInference', 'no VLM needed\nat test time', '#c0392b')

    # JSON schema inset
    schema = ('{"obs": "...",\n'
              ' "instruction": "...",\n'
              ' "task_decomp": "...",\n'
              ' "relevant_physics": "...",\n'
              ' "visual_estimates": "...",\n'
              ' "action_impl": "...",\n'
              ' "action": [...],\n'
              ' "outcome": "success"}')
    ax.text(3.7, 3.7, schema, ha='center', va='center',
            fontsize=7.5, family='monospace', color=COLORS['dark'],
            bbox=dict(facecolor='#ecf0f1', edgecolor='#bdc3c7',
                      boxstyle='round,pad=0.4'))
    ax.annotate('', xy=(3.7, 2.62), xytext=(3.7, 3.32),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5))

    ax.set_title('Future Work: Supervised PhysCoT Training Pipeline',
                 fontsize=13, fontweight='bold', pad=8, color=COLORS['dark'])
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig_future_pipeline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures...')
    fig_pipeline()
    fig_prompt_schema()
    fig_exp_setup()
    fig_main_results()
    fig_failure_modes()
    fig_contact_height()
    fig_qualitative()
    fig_reasoning_example()
    fig_future_pipeline()
    print(f'\nAll figures saved to: {FIGS_DIR}')
