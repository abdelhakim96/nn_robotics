import numpy as np
import torch
import os
import math
import numpy as np # Ensure numpy is imported
from trajectory_generators import ( # Import new trajectory functions
    generate_circular_trajectory,
    generate_lemniscate_trajectory,
    generate_sinusoidal_hover,
    generate_helix_trajectory,
    generate_accelerating_line
)

# --- PID Controller Structure ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self._prev_error = 0
        self._integral = 0

    def calculate(self, process_variable, dt):
        error = self.setpoint - process_variable
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        self._prev_error = error
        return output

# Placeholder for a more complex attitude/position controller
# This needs significant development based on drone dynamics and state representation
# For now, it will just return some basic values based on position error
def simple_drone_controller(current_state, target_pos, dt):
    # Basic PID for altitude (z) control -> Fz
    # Basic PID for x, y position -> target pitch, roll
    # Basic PID for yaw (optional) -> target yaw rate (or directly psi)

    # Example: Simplified Altitude Control (adjust Kp, Ki, Kd)
    z_pid = PIDController(Kp=5.0, Ki=0.1, Kd=0.5, setpoint=target_pos[2])
    current_z = current_state[2]
    # Need base thrust to counteract gravity (m*g)
    base_thrust = 3.8 * 9.81
    fz_adjustment = z_pid.calculate(current_z, dt)
    Fz = np.clip(base_thrust + fz_adjustment, 1, 10) # Clip thrust

    # Example: Simplified XY control -> Target Roll/Pitch (very basic)
    # This needs proper conversion from position error to attitude commands
    target_roll = np.clip(0.1 * (target_pos[1] - current_state[1]), -0.2, 0.2) # Phi command based on y error
    target_pitch = np.clip(-0.1 * (target_pos[0] - current_state[0]), -0.2, 0.2) # Theta command based on x error (negative sign typical)

    # Example: Hold zero yaw
    target_yaw = 0.0 # Psi command

    # Control vector: [phi, theta, psi, Fz] - Note: Using target attitudes directly here is a simplification.
    # A real controller would likely control rates or use a more complex attitude controller.
    u = np.array([target_roll, target_pitch, target_yaw, Fz], dtype=np.float32)
    return u


def generate_lemniscate_trajectory(t, scale=2.0, speed=0.3, center_x=0, center_y=0, z=1.0):
    """Generates points on a lemniscate (figure-eight) trajectory."""
    angle = speed * t
    x_ref = center_x + scale * np.cos(angle) / (1 + np.sin(angle)**2)
    y_ref = center_y + scale * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle)**2)
    z_ref = z
    return np.array([x_ref, y_ref, z_ref])

def generate_sinusoidal_hover(t, axis='z', amplitude=0.5, frequency=0.5, hover_pos=[0,0,1]):
    """Generates a hover trajectory with sinusoidal motion along one axis."""
    pos = np.array(hover_pos, dtype=np.float32)
    offset = amplitude * np.sin(2 * np.pi * frequency * t)
    if axis == 'x':
        pos[0] += offset
    elif axis == 'y':
        pos[1] += offset
    else: # Default to z
        pos[2] += offset
    return pos


def drone_dynamics(x, u, dt):
    # States: x, y, z, phi, theta, psi, u, v, w, p_rate, q_rate, r_rate, Fx_dist, Fy_dist, Fz_dist (15 states)
    # Controls: Fx_cmd, Fy_cmd, Fz_cmd, tau_psi_cmd (Example - assuming force/torque commands now, adjust controller)
    # OR keep controls as target phi, theta, psi, Fz and use them in the controller, not directly in dynamics.
    # Let's assume the controller outputs target phi, theta, psi, Fz as before, but dynamics uses state phi, theta, psi.
    # We need angular accelerations to integrate p_rate, q_rate, r_rate, but the simple model assumes they are constant.
    # Let's stick to the original simple model's assumption for now: p, q, r rates are constant within a step.
    # We WILL integrate phi, theta, psi based on p, q, r.

    # Unpack state (15 elements)
    x_pos, y_pos, z_pos, phi, theta, psi, u_vel, v_vel, w_vel, p_rate, q_rate, r_rate, Fx_dist, Fy_dist, Fz_dist = x
    # Unpack control (assuming controller output this)
    phi_cmd, theta_cmd, psi_cmd, Fz_cmd = u # Note: These cmds might not be directly used if p,q,r are state

    m = 3.8
    g = 9.81

    # Kinematics (Angle derivatives) - handle potential division by zero for tan(theta) near +/- pi/2
    cos_theta = np.cos(theta)
    if abs(cos_theta) < 1e-6: # Avoid division by zero
        cos_theta = 1e-6 * np.sign(cos_theta)
    tan_theta = np.tan(theta)
    phi_dot = p_rate + q_rate * np.sin(phi) * tan_theta + r_rate * np.cos(phi) * tan_theta
    theta_dot = q_rate * np.cos(phi) - r_rate * np.sin(phi)
    psi_dot = q_rate * np.sin(phi) / cos_theta + r_rate * np.cos(phi) / cos_theta

    # Dynamics (Position derivatives - use state angles)
    x_dot = (np.cos(theta) * np.cos(psi)) * u_vel + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.sin(psi) * np.cos(phi)) * v_vel + \
            (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w_vel
    y_dot = (np.cos(theta) * np.sin(psi)) * u_vel + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(psi) * np.cos(phi)) * v_vel + \
            (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w_vel
    z_dot = (-np.sin(theta)) * u_vel + (np.sin(phi) * np.cos(theta)) * v_vel + (np.cos(phi) * np.cos(theta)) * w_vel

    # Dynamics (Velocity derivatives - use state angles and rates)
    u_dot = r_rate * v_vel - q_rate * w_vel + g * np.sin(theta) + Fx_dist
    v_dot = p_rate * w_vel - r_rate * u_vel - g * np.sin(phi) * np.cos(theta) + Fy_dist
    # Use Fz_cmd from control input for vertical acceleration
    w_dot = q_rate * u_vel - p_rate * v_vel - g * np.cos(phi) * np.cos(theta) + (1 / m) * (Fz_cmd) + Fz_dist

    # Check for NaN after velocity calculations
    if np.isnan(u_dot) or np.isnan(v_dot) or np.isnan(w_dot):
        print("Warning: NaN detected in velocity derivatives. Returning original state.")
        return x # Return original state to avoid propagating NaN

    # Integrate state using Euler method
    x_next = x.copy()
    x_next[0] = x_pos + x_dot * dt       # x
    x_next[1] = y_pos + y_dot * dt       # y
    x_next[2] = z_pos + z_dot * dt       # z
    x_next[3] = phi + phi_dot * dt       # phi
    x_next[4] = theta + theta_dot * dt   # theta
    x_next[5] = psi + psi_dot * dt       # psi
    x_next[6] = u_vel + u_dot * dt       # u
    x_next[7] = v_vel + v_dot * dt       # v
    x_next[8] = w_vel + w_dot * dt       # w
    # Assuming angular rates and disturbances are constant for this step (as per original simple model)
    x_next[9] = p_rate
    x_next[10] = q_rate
    x_next[11] = r_rate
    x_next[12] = Fx_dist
    x_next[13] = Fy_dist
    x_next[14] = Fz_dist

    return x_next

if __name__ == '__main__':
    # Create directory for saving data
    data_dir = "drone_data"
    os.makedirs(data_dir, exist_ok=True)

    # Parameters - Increased dataset size and length
    dt = 0.01
    num_train_trajectories = 1000 # Increased number
    num_dev_trajectories = 100   # Increased number
    trajectory_length = 400      # Increased length
    N_coll = 10

    # Initial state (15 elements: pos, att, lin_vel, ang_vel, dist)
    x0 = np.zeros(15, dtype=np.float32)
    x0[2] = 0.1 # Start slightly above ground

    # Generate training data
    all_training_x = []
    all_training_u = []
    all_training_t_coll = []
    all_training_time = []

    print(f"--- Generating {num_train_trajectories} Training Trajectories ---")
    for i in range(num_train_trajectories):
        x = x0.copy() + np.random.normal(0, 0.01, x0.shape) # Add small noise to initial state
        x[3:6] = 0 # Ensure zero initial angles
        x[9:12] = 0 # Ensure zero initial rates
        current_trajectory_x = [x]
        current_trajectory_u = []
        current_trajectory_t_coll = []
        current_trajectory_time = [0.0] # Start time at 0

        # --- Select Trajectory Type and Parameters ---
        traj_type = np.random.choice(['circle', 'lemniscate', 'sine_hover_z', 'sine_hover_x', 'sine_hover_y', 'helix', 'line'])
        speed_factor = np.random.uniform(0.5, 1.5) # Wider speed variation
        radius = np.random.uniform(1.0, 3.0)
        scale = np.random.uniform(1.5, 3.5)
        amplitude = np.random.uniform(0.2, 0.8)
        frequency = np.random.uniform(0.2, 0.8)
        pitch = np.random.uniform(0.3, 1.0)
        clockwise = np.random.choice([True, False])
        accel = np.random.uniform(0.1, 0.5)
        line_dir = np.random.rand(3) - 0.5 # Random direction vector
        line_dir[2] = np.random.uniform(0, 0.3) # Limit z direction change for lines

        print(f"Generating training trajectory {i+1}/{num_train_trajectories}: Type={traj_type}, SpeedFactor={speed_factor:.2f}")

        for j in range(trajectory_length - 1):
            current_time = j * dt

            # --- Get Target Position from Trajectory Function ---
            if traj_type == 'circle':
                target_pos = generate_circular_trajectory(current_time, radius=radius, speed=0.4*speed_factor, z_func=lambda t: 1.0 + 0.2*np.sin(0.5*t*speed_factor), clockwise=clockwise)
            elif traj_type == 'lemniscate':
                target_pos = generate_lemniscate_trajectory(current_time, scale=scale, speed=0.25*speed_factor, z=1.0)
            elif traj_type == 'sine_hover_z':
                 target_pos = generate_sinusoidal_hover(current_time, axis='z', amplitude=amplitude, frequency=frequency*speed_factor, hover_pos=[0,0,1])
            elif traj_type == 'sine_hover_x':
                 target_pos = generate_sinusoidal_hover(current_time, axis='x', amplitude=amplitude, frequency=frequency*speed_factor, hover_pos=[0,0,1])
            elif traj_type == 'sine_hover_y':
                 target_pos = generate_sinusoidal_hover(current_time, axis='y', amplitude=amplitude, frequency=frequency*speed_factor, hover_pos=[0,0,1])
            elif traj_type == 'helix':
                 target_pos = generate_helix_trajectory(current_time, radius=radius, speed=0.3*speed_factor, pitch=pitch, z_start=0.1, clockwise=clockwise)
            elif traj_type == 'line':
                 target_pos = generate_accelerating_line(current_time, start_pos=[0,0,0.5], direction=line_dir, initial_speed=0.1*speed_factor, acceleration=accel*speed_factor)
            else: # Default to circle
                 target_pos = generate_circular_trajectory(current_time, radius=2.0, speed=0.5*speed_factor)

            # --- Calculate Control Input using Controller ---
            u = simple_drone_controller(x, target_pos, dt)

            # --- Simulate Drone Dynamics ---
            x_next = drone_dynamics(x, u, dt)

            # --- Store Data ---
            # Only store if simulation step is valid (e.g., not NaN) - basic check
            if not np.isnan(x_next).any():
                current_trajectory_x.append(x_next)
                current_trajectory_u.append(u)
            else:
                print(f"Warning: NaN detected in state at step {j} for trajectory {i}. Stopping trajectory early.")
                break # Stop this trajectory if it becomes unstable

            # Generate t_coll (collocation times) - Ensure t_coll length matches U length (trajectory_length - 1)

            # Generate t_coll (collocation times)
            t_coll = np.random.uniform(0, dt, N_coll).astype(np.float32)
            current_trajectory_t_coll.append(t_coll)

            # Generate time (linearly spaced) - Matches U and t_coll length
            current_trajectory_time.append(dt * (j + 1))

            x = x_next.copy() # Update state for next step

        # Only proceed if trajectory has sufficient length
        if len(current_trajectory_x) < 2:
            print(f"Warning: Trajectory {i} too short. Skipping.")
            continue

        # Convert lists to numpy arrays
        # X has length N, U has length N-1, t_coll has length N-1, time has length N
        current_trajectory_x_np = np.array(current_trajectory_x, dtype=np.float32)
        current_trajectory_u_np = np.array(current_trajectory_u, dtype=np.float32)
        current_trajectory_t_coll_np = np.array(current_trajectory_t_coll, dtype=np.float32)
        current_trajectory_time_np = np.array(current_trajectory_time, dtype=np.float32)

        # Ensure consistent lengths for saving (X: N, U: N, t_coll: N-1, time: N)
        # Pad U to length N
        if current_trajectory_u_np.shape[0] == current_trajectory_x_np.shape[0] - 1:
             current_trajectory_u_np = np.concatenate((current_trajectory_u_np, current_trajectory_u_np[-1:, :]), axis=0)

        # t_coll should remain N-1
        # time should remain N

        # Append to overall lists
        all_training_x.append(current_trajectory_x_np)
        all_training_u.append(current_trajectory_u_np)
        all_training_t_coll.append(current_trajectory_t_coll_np)
        all_training_time.append(current_trajectory_time_np)

    # --- Convert and Save Training Data ---
    if all_training_x: # Check if any valid trajectories were generated
        final_training_x = torch.from_numpy(np.stack(all_training_x))
        final_training_u = torch.from_numpy(np.stack(all_training_u))
        # t_coll might have variable lengths if trajectories stopped early, handle this if needed
        # Pad t_coll to a consistent length
        max_t_coll_len = max(len(t) for t in all_training_t_coll)
        padded_training_t_coll = [np.pad(t, (0, max_t_coll_len - len(t)), 'constant') for t in all_training_t_coll]
        final_training_t_coll = torch.from_numpy(np.stack(padded_training_t_coll))

        final_training_time = torch.from_numpy(np.stack(all_training_time))

        # Final check on shapes before saving
        print(f"Final Training Shapes - X: {final_training_x.shape}, U: {final_training_u.shape}, t_coll: {final_training_t_coll.shape}, time: {final_training_time.shape}")
        # Ensure U length matches X length after padding
        assert final_training_u.shape[1] == final_training_x.shape[1]
        # Ensure t_coll length is consistent
        assert final_training_t_coll.shape[1] == max_t_coll_len
        # Ensure time length matches X length
        assert final_training_time.shape[1] == final_training_x.shape[1]

        torch.save({'X': final_training_x, 'U': final_training_u, 't_coll': final_training_t_coll, 'time': final_training_time}, os.path.join(data_dir, "training_set.pt"))
        print(f"Saved training_set.pt")
    else:
        print("No valid training trajectories generated.")

    # Generate dev data
    all_dev_x = []
    all_dev_u = []
    all_dev_t_coll = []
    all_dev_time = []

    print(f"--- Generating {num_dev_trajectories} Dev Trajectories ---")
    for i in range(num_dev_trajectories):
        x = x0.copy() + np.random.normal(0, 0.01, x0.shape) # Add small noise to initial state
        x[3:6] = 0 # Ensure zero initial angles
        x[9:12] = 0 # Ensure zero initial rates
        current_trajectory_x = [x]
        current_trajectory_u = []
        current_trajectory_t_coll = []
        current_trajectory_time = [0.0]

        # --- Select Trajectory Type and Parameters ---
        # Use a different random seed or ensure variety compared to training set
        traj_type = np.random.choice(['circle', 'lemniscate', 'sine_hover_z', 'sine_hover_x', 'sine_hover_y', 'helix', 'line'])
        speed_factor = np.random.uniform(0.6, 1.4) # Slightly different range
        radius = np.random.uniform(1.2, 2.8)
        scale = np.random.uniform(1.8, 3.2)
        amplitude = np.random.uniform(0.3, 0.7)
        frequency = np.random.uniform(0.25, 0.75)
        pitch = np.random.uniform(0.4, 0.9)
        clockwise = np.random.choice([True, False])
        accel = np.random.uniform(0.15, 0.45)
        line_dir = np.random.rand(3) - 0.5
        line_dir[2] = np.random.uniform(0, 0.25)

        print(f"Generating dev trajectory {i+1}/{num_dev_trajectories}: Type={traj_type}, SpeedFactor={speed_factor:.2f}")


        for j in range(trajectory_length - 1):
            current_time = j * dt

            # --- Get Target Position from Trajectory Function ---
            if traj_type == 'circle':
                 target_pos = generate_circular_trajectory(current_time, radius=radius, speed=0.35*speed_factor, z_func=lambda t: 1.0 + 0.15*np.cos(0.55*t*speed_factor), clockwise=clockwise)
            elif traj_type == 'lemniscate':
                 target_pos = generate_lemniscate_trajectory(current_time, scale=scale, speed=0.22*speed_factor, z=1.1)
            elif traj_type == 'sine_hover_z':
                  target_pos = generate_sinusoidal_hover(current_time, axis='z', amplitude=amplitude, frequency=frequency*speed_factor, hover_pos=[0,0,1])
            elif traj_type == 'sine_hover_x':
                  target_pos = generate_sinusoidal_hover(current_time, axis='x', amplitude=amplitude, frequency=frequency*speed_factor, hover_pos=[0,0,1])
            elif traj_type == 'sine_hover_y':
                  target_pos = generate_sinusoidal_hover(current_time, axis='y', amplitude=amplitude, frequency=frequency*speed_factor, hover_pos=[0,0,1])
            elif traj_type == 'helix':
                 target_pos = generate_helix_trajectory(current_time, radius=radius, speed=0.35*speed_factor, pitch=pitch, z_start=0.1, clockwise=clockwise)
            elif traj_type == 'line':
                 target_pos = generate_accelerating_line(current_time, start_pos=[0,0,0.5], direction=line_dir, initial_speed=0.15*speed_factor, acceleration=accel*speed_factor)
            else: # Default to circle
                 target_pos = generate_circular_trajectory(current_time, radius=1.8, speed=0.45*speed_factor)


            # --- Calculate Control Input using Controller ---
            u = simple_drone_controller(x, target_pos, dt)

            # --- Simulate Drone Dynamics ---
            x_next = drone_dynamics(x, u, dt)

            # --- Store Data ---
            if not np.isnan(x_next).any():
                current_trajectory_x.append(x_next)
                current_trajectory_u.append(u)
            else:
                print(f"Warning: NaN detected in state at step {j} for dev trajectory {i}. Stopping trajectory early.")
                break # Stop this trajectory

            # Generate t_coll (collocation times)
            t_coll = np.random.uniform(0, dt, N_coll).astype(np.float32)
            current_trajectory_t_coll.append(t_coll)

            # Generate time (linearly spaced)
            current_trajectory_time.append(dt * (j + 1))

            x = x_next.copy()

        # Only proceed if trajectory has sufficient length
        if len(current_trajectory_x) < 2:
            print(f"Warning: Dev trajectory {i} too short. Skipping.")
            continue

        # Convert lists to numpy arrays
        current_trajectory_x_np = np.array(current_trajectory_x, dtype=np.float32)
        current_trajectory_u_np = np.array(current_trajectory_u, dtype=np.float32)
        current_trajectory_t_coll_np = np.array(current_trajectory_t_coll, dtype=np.float32)
        current_trajectory_time_np = np.array(current_trajectory_time, dtype=np.float32)

        # Ensure consistent lengths for saving (X: N, U: N, t_coll: N-1, time: N)
        # Pad U to length N
        if current_trajectory_u_np.shape[0] == current_trajectory_x_np.shape[0] - 1:
             current_trajectory_u_np = np.concatenate((current_trajectory_u_np, current_trajectory_u_np[-1:, :]), axis=0)

        # t_coll should remain N-1
        # time should remain N

        all_dev_x.append(current_trajectory_x_np)
        all_dev_u.append(current_trajectory_u_np)
        all_dev_t_coll.append(current_trajectory_t_coll_np)
        all_dev_time.append(current_trajectory_time_np)

    # --- Convert and Save Dev Data ---
    if all_dev_x: # Check if any valid trajectories were generated
        final_dev_x = torch.from_numpy(np.stack(all_dev_x))
        final_dev_u = torch.from_numpy(np.stack(all_dev_u))
        # Handle potential variable lengths for t_coll
        try:
            final_dev_t_coll = torch.from_numpy(np.stack(all_dev_t_coll))
        except ValueError as e:
             print(f"Error stacking t_coll for dev data: {e}. Check trajectory lengths.")
             final_dev_t_coll = torch.tensor([]) # Placeholder

        final_dev_time = torch.from_numpy(np.stack(all_dev_time))

        # Final check on shapes before saving
        print(f"Final Dev Shapes - X: {final_dev_x.shape}, U: {final_dev_u.shape}, t_coll: {final_dev_t_coll.shape}, time: {final_dev_time.shape}")
        # Ensure U length matches X length after padding
        assert final_dev_u.shape[1] == final_dev_x.shape[1]
        # Ensure t_coll length is X length - 1
        assert final_dev_t_coll.shape[1] == final_dev_x.shape[1] - 1
        # Ensure time length matches X length
        assert final_dev_time.shape[1] == final_dev_x.shape[1]

        torch.save({'X': final_dev_x, 'U': final_dev_u, 't_coll': final_dev_t_coll, 'time': final_dev_time}, os.path.join(data_dir, "dev_set.pt"))
        print(f"Saved dev_set.pt")
    else:
        print("No valid dev trajectories generated.")


    print("Drone data generation complete.")
