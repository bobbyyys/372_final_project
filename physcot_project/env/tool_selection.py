"""
Tool Selection Simulation Environment.

2D scene: a robot must move a target object to a goal region.
Two tools are available:
  - Hook: curved end; can reach around obstacles; effective pull force
  - Straight pusher: flat end; can only push directly; blocked by obstacles

Physics model:
  - Object is a disc of radius r_obj
  - Obstacle is a rectangle placed between robot and object
  - Hook can navigate around the obstacle (arc path)
  - Straight tool is blocked if the direct path intersects the obstacle

State: (object position, tool chosen, approach path feasible?)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as mpatches


G = 9.81


class ToolSelectionEnv:
    """
    2D tool-selection environment.

    Layout (all in metres, origin = robot base):
      - Robot base: (0, 0)
      - Obstacle: rectangle centred at (obs_x, obs_y) with (obs_w, obs_h)
      - Target object: disc centred at (obj_x, obj_y)
      - Goal region: circle centred at (goal_x, goal_y) with radius goal_r
      - Tool A (hook):    position at (hook_x, hook_y)
      - Tool B (straight): position at (str_x, str_y)
    """

    def __init__(self,
                 obj_x=0.45, obj_y=0.30,
                 obs_x=0.25, obs_y=0.25,
                 obs_w=0.12, obs_h=0.18,
                 goal_x=0.10, goal_y=0.40,
                 hook_x=0.05, hook_y=0.10,
                 str_x=0.05, str_y=0.25,
                 obj_radius=0.04,
                 goal_radius=0.08,
                 max_steps=200,
                 seed=0):

        self.obj_x0 = obj_x
        self.obj_y0 = obj_y
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.obs_w = obs_w
        self.obs_h = obs_h
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.hook_pos = (hook_x, hook_y)
        self.str_pos = (str_x, str_y)
        self.r_obj = obj_radius
        self.goal_r = goal_radius
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.obj_x = self.obj_x0
        self.obj_y = self.obj_y0
        self.tool_chosen = None
        self.path_feasible = None
        self.task_success = False
        self.step_count = 0
        self.history = []
        self._record('init')
        return self._obs()

    def _record(self, phase='step'):
        self.history.append({
            'obj_x': self.obj_x,
            'obj_y': self.obj_y,
            'tool': self.tool_chosen,
            'phase': phase,
            'step': self.step_count
        })

    def _obs(self):
        return {
            'obj_x': self.obj_x,
            'obj_y': self.obj_y,
            'obs_x': self.obs_x,
            'obs_y': self.obs_y,
            'obs_w': self.obs_w,
            'obs_h': self.obs_h,
            'goal_x': self.goal_x,
            'goal_y': self.goal_y,
            'obj_radius': self.r_obj,
            'goal_radius': self.goal_r,
            'hook_pos': self.hook_pos,
            'str_pos': self.str_pos,
            'step': self.step_count
        }

    def _direct_path_blocked(self, from_xy, to_xy):
        """Return True if straight line from_xy→to_xy is blocked by obstacle."""
        # Obstacle bounding box corners
        ox1 = self.obs_x - self.obs_w / 2
        ox2 = self.obs_x + self.obs_w / 2
        oy1 = self.obs_y - self.obs_h / 2
        oy2 = self.obs_y + self.obs_h / 2

        # Test if segment intersects any of the 4 obstacle edges
        corners = [
            ((ox1, oy1), (ox2, oy1)),
            ((ox2, oy1), (ox2, oy2)),
            ((ox2, oy2), (ox1, oy2)),
            ((ox1, oy2), (ox1, oy1)),
        ]
        for (c1, c2) in corners:
            if _seg_intersect(from_xy, to_xy, c1, c2):
                return True
        return False

    def step(self, tool_choice):
        """
        Choose a tool and attempt to move the object to the goal.

        tool_choice : str — 'hook' or 'straight'

        Returns obs, success, info
        """
        self.tool_chosen = tool_choice
        self.step_count += 1

        # Check whether the straight path from robot to object is blocked
        robot_pos = (0.0, 0.0)
        obj_pos = (self.obj_x, self.obj_y)

        direct_blocked = self._direct_path_blocked(robot_pos, obj_pos)

        # === Feasibility logic ===
        if tool_choice == 'hook':
            # Hook can always navigate around obstacle via arc path
            # Small chance of failure due to awkward geometry
            arc_clearance = self._compute_hook_clearance()
            feasible = arc_clearance > 0.02   # at least 2cm clearance for arc
        elif tool_choice == 'straight':
            # Straight tool requires unobstructed direct path
            feasible = not direct_blocked
        else:
            feasible = False

        # === Simulate motion toward goal if feasible ===
        self.path_feasible = feasible

        if feasible:
            # Animate object moving toward goal
            n_move = 80
            start = np.array([self.obj_x, self.obj_y])
            end = np.array([self.goal_x, self.goal_y])
            for i in range(n_move):
                t = (i + 1) / n_move
                pos = (1 - t) * start + t * end
                self.obj_x, self.obj_y = pos
                self._record('moving')

            # Check if object reached goal
            dist_to_goal = np.hypot(self.obj_x - self.goal_x,
                                    self.obj_y - self.goal_y)
            self.task_success = dist_to_goal < self.goal_r
        else:
            # Object doesn't move; record a few static frames
            for _ in range(30):
                self._record('blocked')
            self.task_success = False

        info = {
            'tool_chosen': tool_choice,
            'direct_blocked': direct_blocked,
            'path_feasible': feasible,
            'success': self.task_success,
            'final_obj_pos': (self.obj_x, self.obj_y),
            'dist_to_goal': np.hypot(self.obj_x - self.goal_x,
                                     self.obj_y - self.goal_y),
            'correct_tool': 'hook' if direct_blocked else 'straight'
        }

        return self._obs(), self.task_success, info

    def _compute_hook_clearance(self):
        """
        Estimate clearance for hook arc around obstacle.
        Returns approximate clearance in metres.
        """
        # Arc goes around the obstacle on the "open" side
        # Open side is determined by which side of obstacle has more space
        # Simplified: check if left side (lower y) or right side (upper y) is free
        arc_radius = max(self.obs_w, self.obs_h) * 0.8 + self.r_obj + 0.02
        # Check minimum distance from arc path to obstacle corners
        # Approximate as obstacle half-diagonal
        obs_diag = np.hypot(self.obs_w, self.obs_h) / 2
        clearance = arc_radius - obs_diag
        return max(clearance, 0.0)

    def render_video(self, save_path, title='Tool Selection'):
        """Render simulation history as MP4 video."""
        fig, ax = plt.subplots(figsize=(6, 6))

        def draw_frame(i):
            ax.clear()
            ax.set_xlim(-0.05, 0.65)
            ax.set_ylim(-0.05, 0.65)
            ax.set_aspect('equal')
            ax.set_facecolor('#f8f8f8')
            ax.grid(True, alpha=0.3)

            rec = self.history[i]

            # Goal region
            goal_circle = plt.Circle((self.goal_x, self.goal_y),
                                     self.goal_r, color='#2ecc71', alpha=0.3, zorder=1)
            ax.add_patch(goal_circle)
            ax.plot(self.goal_x, self.goal_y, 'g*', ms=12, zorder=2)

            # Obstacle
            obs_rect = plt.Rectangle(
                (self.obs_x - self.obs_w / 2, self.obs_y - self.obs_h / 2),
                self.obs_w, self.obs_h,
                color='#e74c3c', alpha=0.7, zorder=3)
            ax.add_patch(obs_rect)
            ax.text(self.obs_x, self.obs_y, 'OBS', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

            # Tool positions
            ax.plot(*self.hook_pos, 'b^', ms=12, zorder=4)
            ax.text(self.hook_pos[0] + 0.02, self.hook_pos[1], 'Hook',
                    fontsize=8, color='blue')
            ax.plot(*self.str_pos, 'ms', ms=12, zorder=4)
            ax.text(self.str_pos[0] + 0.02, self.str_pos[1], 'Straight',
                    fontsize=8, color='purple')

            # Robot base
            ax.plot(0, 0, 'ko', ms=14, zorder=5)
            ax.text(0.01, -0.03, 'Robot', fontsize=8)

            # Target object
            obj_circle = plt.Circle((rec['obj_x'], rec['obj_y']),
                                    self.r_obj, color='#f39c12', alpha=0.9, zorder=6)
            ax.add_patch(obj_circle)
            ax.text(rec['obj_x'], rec['obj_y'], 'T', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

            tool_str = rec['tool'] if rec['tool'] else 'None'
            ax.set_title(f'{title}\nTool: {tool_str} | Phase: {rec["phase"]}',
                         fontsize=10)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')

        n_frames = len(self.history)
        frame_indices = np.linspace(0, n_frames - 1, min(n_frames, 80), dtype=int)

        writer = FFMpegWriter(fps=15, metadata={'title': title})
        with writer.saving(fig, save_path, dpi=100):
            for idx in frame_indices:
                draw_frame(idx)
                writer.grab_frame()

        plt.close(fig)

    def get_scene_description(self):
        """Return text description of current scene for prompts."""
        direct_blocked = self._direct_path_blocked((0, 0),
                                                   (self.obj_x, self.obj_y))
        block_desc = "is blocked by the obstacle" if direct_blocked \
                     else "has a clear straight path"
        return (
            f"A target object (disc, radius {self.r_obj*100:.1f}cm) is at "
            f"position ({self.obj_x*100:.0f}cm, {self.obj_y*100:.0f}cm). "
            f"An obstacle (box {self.obs_w*100:.0f}cm×{self.obs_h*100:.0f}cm) "
            f"is at ({self.obs_x*100:.0f}cm, {self.obs_y*100:.0f}cm). "
            f"The direct path from robot to object {block_desc}. "
            f"Goal region is at ({self.goal_x*100:.0f}cm, {self.goal_y*100:.0f}cm). "
            f"Available tools: Hook (curved end, can reach around obstacles) "
            f"and Straight pusher (flat end, requires unobstructed path). "
            f"Task: move the object to the goal region."
        )


def _seg_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross2d(v, w):
        return v[0] * w[1] - v[1] * w[0]
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    cross = cross2d(d1, d2)
    if abs(cross) < 1e-10:
        return False
    dx = p3[0] - p1[0]
    dy = p3[1] - p1[1]
    t = cross2d((dx, dy), d2) / cross
    u = cross2d((dx, dy), d1) / cross
    return 0 <= t <= 1 and 0 <= u <= 1
