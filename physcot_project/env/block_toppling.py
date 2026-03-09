"""
Block Toppling Simulation Environment.

2D rigid-body physics for a rectangular block that can be pushed over.

Physics model:
  - Block: width w, height h, mass m
  - Initially upright on a flat surface (θ = 0)
  - Robot end-effector applies lateral horizontal force F at height y_c from ground
  - Pivot: bottom edge of block on push side
  - Toppling torque:  τ_top  = F * y_c
  - Restoring torque: τ_rest = m * g * (w/2) * cos(θ)
  - Friction: if F > μ * m * g, block may slide instead of rotate

State: (θ, ω) — block tilt angle and angular velocity
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter
import os


G = 9.81


class BlockToppingEnv:
    """
    2D block-toppling environment.

    Parameters
    ----------
    block_width  : float  (metres)
    block_height : float  (metres)
    block_mass   : float  (kg)
    friction_coef: float  (dimensionless)
    push_force   : float  (N) – maximum push force applied by robot
    dt           : float  (s) – integration step
    max_steps    : int    – rollout horizon
    """

    def __init__(self,
                 block_width=0.08,
                 block_height=0.20,
                 block_mass=0.5,
                 friction_coef=0.5,
                 push_force=5.0,
                 dt=0.01,
                 max_steps=300,
                 seed=0):

        self.w = block_width
        self.h = block_height
        self.m = block_mass
        self.mu = friction_coef
        self.F = push_force
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Moment of inertia about COM (solid rectangular block)
        self.I_com = (self.m / 12.0) * (self.w**2 + self.h**2)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.theta = 0.0       # block tilt angle (rad), 0 = upright
        self.omega = 0.0       # angular velocity (rad/s)
        self.x_block = 0.0    # horizontal position of block base centre
        self.v_block = 0.0    # horizontal velocity (sliding)
        self.contact_made = False
        self.step_count = 0
        self.history = []
        self._record()
        return self._obs()

    # ------------------------------------------------------------------
    def _record(self):
        self.history.append({
            'theta': self.theta,
            'omega': self.omega,
            'x': self.x_block,
            'step': self.step_count
        })

    def _obs(self):
        return {
            'theta': self.theta,
            'omega': self.omega,
            'x_block': self.x_block,
            'block_width': self.w,
            'block_height': self.h,
            'block_mass': self.m,
            'friction': self.mu,
            'step': self.step_count
        }

    # ------------------------------------------------------------------
    def _pivot_distance(self):
        """
        Distance from COM to pivot edge (bottom-right corner of block in
        the rotated frame).  Returns r, the moment arm length.
        """
        return 0.5 * np.sqrt(self.w**2 + self.h**2)

    # ------------------------------------------------------------------
    def step(self, contact_height_frac, force_direction_sign=1.0):
        """
        Apply one push action and integrate until force is removed or
        block has clearly toppled / settled.

        contact_height_frac : float in [0, 1] — fractional height of
                              contact point on the block (0=base, 1=top)
        force_direction_sign : +1 (push right) or -1 (push left)

        Returns obs, success, info
        """
        y_c = contact_height_frac * self.h   # absolute contact height (m)
        F = self.F * force_direction_sign

        # Friction limit
        F_friction_max = self.mu * self.m * G

        results = []
        toppled = False
        slid_only = False

        for _ in range(self.max_steps):
            self.step_count += 1

            if abs(self.theta) > np.pi / 2.0:
                # Block has fallen flat
                toppled = True
                break

            # --- Determine if block rotates or slides ---
            # Toppling torque about pivot edge (bottom corner on push side)
            # Pivot offset from base centre: w/2 in the direction of push
            # Torque due to applied force about pivot:
            #   τ_F = F * (y_c * cos(θ) - sign * (w/2) * sin(θ))   [approx]
            # Restoring torque from gravity about pivot:
            #   τ_g = -m*g*(w/2)*cos(θ)   [restoring]
            #
            # Simplified: treat pivot at base edge

            cos_t = np.cos(self.theta)
            sin_t = np.sin(self.theta)

            # Effective torque
            tau_push = F * (y_c * cos_t - (self.w / 2.0) * sin_t * np.sign(F))
            tau_grav = -self.m * G * (self.w / 2.0) * cos_t * np.sign(F)

            # Moment of inertia about pivot (parallel axis theorem)
            I_pivot = self.I_com + self.m * (self.w**2 / 4.0 + self.h**2 / 4.0)

            tau_net = tau_push + tau_grav

            # Check if frictional resistance allows sliding vs rotating
            # Normal force at base ~ m*g (simplified)
            # Sliding: F > mu * m * g and torque is insufficient
            if abs(F) > F_friction_max and abs(tau_push) < abs(tau_grav) * 0.5:
                # Block tends to slide
                a_slide = (abs(F) - F_friction_max) / self.m * np.sign(F)
                self.v_block += a_slide * self.dt
                self.x_block += self.v_block * self.dt
                slid_only = True
            else:
                # Block rotates about pivot
                alpha = tau_net / I_pivot
                self.omega += alpha * self.dt
                self.theta += self.omega * self.dt
                # Apply damping
                self.omega *= 0.99

            self._record()

            if abs(self.theta) > np.radians(60):
                toppled = True
                break

        # Continue integrating under gravity after push (free fall)
        if not toppled and self.theta > np.radians(10):
            for _ in range(200):
                cos_t = np.cos(self.theta)
                tau_grav_only = self.m * G * (self.w / 2.0) * cos_t * np.sign(self.theta)
                I_pivot = self.I_com + self.m * (self.w**2 / 4.0 + self.h**2 / 4.0)
                alpha = tau_grav_only / I_pivot
                self.omega += alpha * self.dt
                self.theta += self.omega * self.dt
                self.omega *= 0.98
                self._record()
                if abs(self.theta) > np.radians(60):
                    toppled = True
                    break

        success = toppled
        max_tilt_deg = np.degrees(max(abs(r['theta']) for r in self.history))

        info = {
            'toppled': toppled,
            'slid_only': slid_only,
            'max_tilt_deg': max_tilt_deg,
            'final_theta_deg': np.degrees(self.theta),
            'contact_height_frac': contact_height_frac,
            'y_c': y_c,
            'steps': self.step_count
        }

        return self._obs(), success, info

    # ------------------------------------------------------------------
    def render_video(self, save_path, title='Block Toppling'):
        """Render simulation history as an MP4 video."""
        fig, ax = plt.subplots(figsize=(6, 5))

        table_y = 0.0
        ground_x = (-0.5, 0.5)

        def draw_frame(i):
            ax.clear()
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(-0.05, 0.35)
            ax.set_aspect('equal')
            ax.set_facecolor('#f0f0f0')

            # Ground
            ax.axhline(table_y, color='#8B4513', lw=2)
            ax.fill_between([-0.4, 0.4], -0.05, 0, color='#c8a97a', alpha=0.5)

            rec = self.history[i]
            theta = rec['theta']
            x_b = rec['x']

            # Block corners relative to pivot (bottom-left corner when theta=0)
            # Block base centre at (x_b, 0)
            cx = x_b
            cy = self.h / 2.0

            # Rotate corners about base-centre (bottom of block, at x_b, 0)
            corners_local = [
                (-self.w / 2, 0), (self.w / 2, 0),
                (self.w / 2, self.h), (-self.w / 2, self.h)
            ]
            pivot = (x_b, 0.0)
            corners = []
            for px, py in corners_local:
                rx = pivot[0] + px * np.cos(theta) - py * np.sin(theta)
                ry = pivot[1] + px * np.sin(theta) + py * np.cos(theta)
                corners.append((rx, ry))

            poly = plt.Polygon(corners, closed=True,
                               facecolor='#4a90d9', edgecolor='#2c5f8a', lw=2)
            ax.add_patch(poly)

            # COM marker
            com_x = pivot[0] + (0) * np.cos(theta) - (self.h / 2) * np.sin(theta)
            com_y = pivot[1] + (0) * np.sin(theta) + (self.h / 2) * np.cos(theta)
            ax.plot(com_x, com_y, 'r+', ms=10, mew=2)

            ax.set_title(f'{title} — step {rec["step"]:03d} | θ={np.degrees(theta):.1f}°',
                         fontsize=11)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.grid(True, alpha=0.3)

        # Subsample frames for video
        n_frames = len(self.history)
        frame_indices = np.linspace(0, n_frames - 1, min(n_frames, 100), dtype=int)

        writer = FFMpegWriter(fps=20, metadata={'title': title})
        with writer.saving(fig, save_path, dpi=100):
            for idx in frame_indices:
                draw_frame(idx)
                writer.grab_frame()

        plt.close(fig)

    # ------------------------------------------------------------------
    def get_scene_description(self):
        """Return text description of current scene (used for prompts)."""
        aspect = self.h / self.w
        stability = "tall and narrow (easy to topple)" if aspect > 2 else \
                    "moderate aspect ratio" if aspect > 1.5 else "wide and stable"
        return (
            f"A rectangular block stands upright on a flat surface. "
            f"Block dimensions: {self.w*100:.1f}cm wide x {self.h*100:.1f}cm tall. "
            f"Mass: {self.m:.2f}kg. "
            f"Surface friction coefficient: {self.mu:.2f}. "
            f"The block is {stability} (aspect ratio {aspect:.1f}). "
            f"COM is at height {self.h/2*100:.1f}cm from the ground. "
            f"Goal: push the block over so it falls (tilt > 60°)."
        )
