import os
import json
import random

def generate_physcot_reasoning_trace(task_type: str, context: dict) -> str:
    """
    Simulates a Vision-Language Model (VLM) generating the 4-step PhysCoT reasoning trace
    based on the scene context.
    """
    if task_type == "block_toppling":
        w = context.get('width_cm', 10)
        h = context.get('height_cm', 20)
        m = context.get('mass_kg', 0.5)
        mu = context.get('friction', 0.5)
        f_push = 5.0
        
        # Physics approximation (from Eq 4)
        yc_star = (m * 9.8 * w * 0.01) / (2 * f_push)
        target_h = max(yc_star + 0.15, 0.65)
        
        return f"""Step A. Task decomposition:
  Goal: topple the upright block.
  Sub-goal: create torque > restoring torque.

Step B. Relevant physics:
  τ_push = F·y_c, τ_crit = mg(w/2)
  Topple if y_c > mgw/(2F) = {yc_star:.2f}m
  Friction: F_slide = μmg

Step C. Visual physical estimates:
  Aspect ratio h/w = {h/w:.2f}
  Best contact zone: [{yc_star+0.15:.2f}, 0.88] × h

Step D. Action implication:
  Contact at {target_h*100:.0f}% height (above y_c*)
  Lateral horizontal push.
  τ_push > τ_crit: topple expected."""

    elif task_type == "tool_selection":
        blocked = context.get('path_blocked', True)
        tool = "HOOK" if blocked else "STRAIGHT"
        return f"""Step A. Task decomposition:
  Goal: move object to goal region.
  Sub-goal: select tool with correct affordance.

Step B. Relevant physics:
  Hook: curved → arc path around obstacle.
  Straight: direct axis push only.

Step C. Visual physical estimates:
  Direct path: {'BLOCKED' if blocked else 'CLEAR'}

Step D. Action implication:
  Select {tool}.
  Approach from open lateral side."""
  
    return "No reasoning generated."

def create_mock_dataset(num_samples=100, out_dir="data"):
    """
    Generates a mock jsonl dataset mapping images and instructions to PhysCoT reasoning and actions.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "physcot_training_data.jsonl")
    
    with open(out_file, 'w') as f:
        for i in range(num_samples):
            task_type = random.choice(["block_toppling", "tool_selection"])
            
            if task_type == "block_toppling":
                context = {
                    "width_cm": random.randint(5, 12),
                    "height_cm": random.randint(15, 25),
                    "mass_kg": random.uniform(0.3, 0.8),
                    "friction": random.uniform(0.3, 0.7)
                }
                instruction = "Push the block over."
                action_str = f"push_contact_h_{random.uniform(0.65, 0.88):.2f}"
            else:
                blocked = random.choice([True, False])
                context = {"path_blocked": blocked}
                instruction = "Move the object past the obstacle."
                action_str = "select_hook" if blocked else "select_straight"
                
            reasoning = generate_physcot_reasoning_trace(task_type, context)
            
            sample = {
                "image_path": f"fake_images/img_{i:04d}.jpg",
                "instruction": instruction,
                "physcot_reasoning": reasoning,
                "action": action_str
            }
            f.write(json.dumps(sample) + "\n")
            
    print(f"Generated {num_samples} PhysCoT training samples at {out_file}")

if __name__ == "__main__":
    create_mock_dataset()
