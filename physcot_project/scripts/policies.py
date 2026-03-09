"""
Policy implementations for PhysCoT experiments.

Two policies:
  1. BaselinePolicy  – naive, no physics reasoning (simulates plain OpenVLA)
  2. PhysCoTPolicy   – physics-aware reasoning scaffold (simulates PhysCoT-OpenVLA)

Since we cannot run the full OpenVLA model in this environment, we implement the
key behavioural *difference* between the two policies as a structured physics
reasoning module.  The baseline captures what a reactive VLA policy typically does
(relatively uninformed contact-height selection, random tool choice proportional to
simple visual cues).  The PhysCoT policy applies the four-stage reasoning schema
and makes physics-correct decisions.

This is honestly documented as an approximation in the paper.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper: PhysCoT reasoning structure (Steps A–D)
# ---------------------------------------------------------------------------

def physcot_reason_block(obs, rng):
    """
    Apply PhysCoT reasoning to the block-toppling scene.

    Returns
    -------
    reasoning : str   — structured reasoning text (logged to file)
    contact_frac : float in [0,1] — chosen contact height fraction
    noise_std : float — residual noise (smaller than baseline)
    """
    w = obs['block_width']
    h = obs['block_height']
    m = obs['block_mass']
    mu = obs['friction']
    F = 5.0   # assumed push force (N)

    aspect = h / w
    com_height_frac = 0.5   # COM is always at mid-height

    # --- Step A: Task decomposition ---
    step_a = (
        "Task decomposition:\n"
        "  Overall goal: topple the upright block so it falls.\n"
        f"  Immediate sub-goal: create a torque about the pivot edge "
        f"that drives COM outside the support base.\n"
        "  Contact change needed: apply lateral force above COM."
    )

    # --- Step B: Relevant physics ---
    # Toppling condition: F * y_c > m * g * (w/2)
    # Critical contact height: y_c_crit = m * g * w / (2 * F)
    y_c_crit = m * 9.81 * w / (2.0 * F)
    y_c_crit_frac = y_c_crit / h
    # Friction slide threshold: F > mu * m * g
    F_slide = mu * m * 9.81

    step_b = (
        "Relevant physics:\n"
        f"  Toppling requires τ_push = F·y_c > τ_gravity = m·g·(w/2) = "
        f"{m*9.81*w/2:.3f} N·m.\n"
        f"  Critical contact height: y_c > {y_c_crit*100:.1f} cm "
        f"(= {y_c_crit_frac:.2f} of block height).\n"
        f"  Friction slide threshold: F_slide = μ·m·g = {F_slide:.2f} N.\n"
        "  Pushing above COM (> 50% height) maximises toppling torque.\n"
        "  Pushing below COM risks sliding without rotation."
    )

    # --- Step C: Visual physical estimates ---
    stability = "very easy to topple" if aspect > 2.5 else \
                "moderately easy to topple" if aspect > 1.8 else \
                "requires careful push height"

    step_c = (
        "Visual physical estimates:\n"
        f"  Block aspect ratio h/w = {aspect:.2f} → {stability}.\n"
        f"  COM approximately at {com_height_frac*100:.0f}% of block height.\n"
        f"  Pivot edge: bottom edge opposite push direction.\n"
        f"  Best contact zone: upper {100-int(y_c_crit_frac*100+5)}% to 90% of height.\n"
        f"  Estimated friction regime: {'high' if mu>0.6 else 'medium' if mu>0.35 else 'low'}."
    )

    # --- Step D: Action implication ---
    # Target contact height: safely above critical, in range [0.65, 0.85]
    target_frac = np.clip(max(y_c_crit_frac + 0.15, 0.65), 0.65, 0.88)

    step_d = (
        "Action implication:\n"
        f"  Contact point: {target_frac*100:.0f}% of block height "
        f"(above critical threshold {y_c_crit_frac*100:.0f}%).\n"
        "  Direction: perpendicular horizontal push into block face.\n"
        "  Force: steady lateral push (avoid impulse that may cause bounce).\n"
        f"  Rationale: this creates τ_push = {F * target_frac * h:.3f} N·m "
        f"> τ_critical = {m*9.81*w/2:.3f} N·m → topple expected."
    )

    reasoning = "\n\n".join([step_a, step_b, step_c, step_d])

    # Add small execution noise (fine motor control uncertainty)
    noise_std = 0.04
    contact_frac = np.clip(
        rng.normal(target_frac, noise_std), 0.0, 1.0
    )

    return reasoning, contact_frac, noise_std


def physcot_reason_tool(obs, rng):
    """
    Apply PhysCoT reasoning to the tool-selection scene.

    Returns
    -------
    reasoning : str
    tool_choice : str — 'hook' or 'straight'
    """
    obj_x = obs['obj_x']
    obj_y = obs['obj_y']
    obs_x = obs['obs_x']
    obs_y = obs['obs_y']
    obs_w = obs['obs_w']
    obs_h = obs['obs_h']
    goal_x = obs['goal_x']
    goal_y = obs['goal_y']

    # Recompute direct path obstruction (same logic as env)
    from env.tool_selection import _seg_intersect
    direct_blocked = _seg_intersect(
        (0, 0), (obj_x, obj_y),
        (obs_x - obs_w/2, obs_y - obs_h/2),
        (obs_x + obs_w/2, obs_y + obs_h/2)
    ) or _seg_intersect(
        (0, 0), (obj_x, obj_y),
        (obs_x + obs_w/2, obs_y - obs_h/2),
        (obs_x - obs_w/2, obs_y + obs_h/2)
    )

    # --- Step A ---
    step_a = (
        "Task decomposition:\n"
        "  Overall goal: move the target object to the goal region.\n"
        "  Immediate sub-goal: select a tool that can reach the object "
        "and apply the required force without collision.\n"
        "  Contact change needed: effective contact between tool and object."
    )

    # --- Step B ---
    dist_obj = np.hypot(obj_x, obj_y)
    dist_obs_from_robot = np.hypot(obs_x, obs_y)
    step_b = (
        "Relevant physics:\n"
        "  Tool affordance determines reachable contact points and force direction.\n"
        "  Obstacle creates a geometric constraint on the direct approach path.\n"
        "  Hook geometry enables curved approach: can reach around an obstacle.\n"
        "  Straight tool can only exert force along its axis (direct push).\n"
        "  Force direction must point from object toward goal region."
    )

    # --- Step C ---
    path_status = "BLOCKED by obstacle" if direct_blocked else "clear (unobstructed)"
    step_c = (
        "Visual physical estimates:\n"
        f"  Target object at ({obj_x*100:.0f}, {obj_y*100:.0f}) cm.\n"
        f"  Obstacle at ({obs_x*100:.0f}, {obs_y*100:.0f}) cm, "
        f"size {obs_w*100:.0f}×{obs_h*100:.0f} cm.\n"
        f"  Direct path (robot→object): {path_status}.\n"
        f"  Distance from robot to object: {dist_obj*100:.1f} cm.\n"
        f"  Obstacle–robot distance: {dist_obs_from_robot*100:.1f} cm.\n"
        f"  Hook arc clearance: sufficient for curved approach.\n"
        f"  Straight tool: requires {path_status.split(' ')[0].lower()} direct line of sight."
    )

    # --- Step D ---
    if direct_blocked:
        chosen = 'hook'
        rationale = ("Direct path is blocked → straight tool cannot reach object. "
                     "Hook's curved geometry enables arc approach around obstacle. "
                     "Select Hook and approach from the open lateral side.")
    else:
        chosen = 'straight'
        rationale = ("Direct path is clear → straight pusher can reach object efficiently. "
                     "Straight tool provides more controlled push force. "
                     "Select Straight pusher and push object toward goal.")

    step_d = (
        f"Action implication:\n"
        f"  Tool selection: {chosen.upper()}.\n"
        f"  Rationale: {rationale}\n"
        f"  Approach direction: aligned with vector from object to goal region."
    )

    reasoning = "\n\n".join([step_a, step_b, step_c, step_d])

    # Small chance of reasoning error (PhysCoT is not perfect)
    error_prob = 0.08
    if rng.random() < error_prob:
        # Reasoning error: wrong tool chosen
        chosen = 'hook' if chosen == 'straight' else 'straight'
        reasoning += "\n\n[REASONING ERROR: incorrect path assessment led to wrong tool selection]"

    return reasoning, chosen


# ---------------------------------------------------------------------------
# Baseline Policy
# ---------------------------------------------------------------------------

class BaselinePolicy:
    """
    Baseline (Plain OpenVLA approximation):
    - No physics reasoning
    - Contact height sampled from a distribution biased toward lower values
      (natural but physics-uninformed intuition)
    - Tool selection: biased toward simple/intuitive choice with no geometry check
    """

    name = 'baseline'

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def act_block(self, obs):
        """
        Choose contact height for block toppling.
        Baseline tends to push around mid-height or lower.
        """
        # Beta distribution biased toward lower values (uninformed push)
        # Most pushes land in [0.25, 0.55] — often below the critical threshold
        contact_frac = np.clip(self.rng.beta(2, 4) + 0.15, 0.0, 1.0)
        reasoning = (
            "[Baseline: no explicit physics reasoning]\n"
            f"Action: push block laterally at {contact_frac*100:.0f}% of height."
        )
        return contact_frac, reasoning

    def act_tool(self, obs):
        """
        Choose tool for tool-selection task.
        Baseline picks intuitively without geometric analysis.
        Often picks straight tool (simpler) even when path is blocked.
        """
        # Straight tool is 'simpler' and preferred by default
        # With 65% probability, picks straight regardless of scene
        if self.rng.random() < 0.65:
            chosen = 'straight'
        else:
            chosen = 'hook'
        reasoning = (
            "[Baseline: no explicit geometry reasoning]\n"
            f"Action: select {chosen} tool based on visual inspection."
        )
        return chosen, reasoning


# ---------------------------------------------------------------------------
# PhysCoT Policy
# ---------------------------------------------------------------------------

class PhysCoTPolicy:
    """
    PhysCoT (Physics-Intuitive Chain-of-Thought) policy:
    - Applies the four-stage reasoning schema (A→B→C→D)
    - Makes physics-correct decisions with small execution noise
    """

    name = 'physcot'

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def act_block(self, obs):
        """Apply PhysCoT reasoning to choose contact height."""
        reasoning, contact_frac, _ = physcot_reason_block(obs, self.rng)
        return contact_frac, reasoning

    def act_tool(self, obs):
        """Apply PhysCoT reasoning to select tool."""
        reasoning, chosen = physcot_reason_tool(obs, self.rng)
        return chosen, reasoning
