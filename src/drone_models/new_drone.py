import torch
# Ensure torch is imported if not already (redundant import removed)
from src.drone_model import DroneModel

class NewDrone(DroneModel):
    # Accept mass and inertia, provide defaults if appropriate
    def __init__(self, mass=3.8, inertia=torch.eye(3)): # Example default inertia
        super().__init__(mass, inertia) # Pass arguments to parent
        # self.m is now inherited from DroneModel as self.mass
        self.g = 9.81  # m/s^2
        self.dt = 0.01 # Time step (should match data generation)

    def forward(self, state, control_input):
        # state shape: [batch_size, 15]
        # control_input shape: [batch_size, 4] (phi_cmd, theta_cmd, psi_cmd, Fz_cmd)

        # Unpack state (15 elements)
        x_pos   = state[..., 0]
        y_pos   = state[..., 1]
        z_pos   = state[..., 2]
        phi     = state[..., 3] # Use state phi
        theta   = state[..., 4] # Use state theta
        psi     = state[..., 5] # Use state psi
        u_vel   = state[..., 6]
        v_vel   = state[..., 7]
        w_vel   = state[..., 8]
        p_rate  = state[..., 9]
        q_rate  = state[..., 10]
        r_rate  = state[..., 11]
        # Fx_dist = state[..., 12] # Removed
        # Fy_dist = state[..., 13] # Removed
        # Fz_dist = state[..., 14] # Removed

        # Unpack control (Fz_cmd is used in w_dot)
        Fz_cmd  = control_input[..., 3]

        # Kinematics (Angle derivatives)
        cos_theta = torch.cos(theta)
        # Add small epsilon to prevent division by zero, ensure it's on the right device
        epsilon = torch.tensor(1e-6, device=state.device)
        cos_theta_safe = torch.where(torch.abs(cos_theta) < epsilon, epsilon * torch.sign(cos_theta), cos_theta)
        tan_theta = torch.tan(theta)

        phi_dot = p_rate + q_rate * torch.sin(phi) * tan_theta + r_rate * torch.cos(phi) * tan_theta
        theta_dot = q_rate * torch.cos(phi) - r_rate * torch.sin(phi)
        psi_dot = q_rate * torch.sin(phi) / cos_theta_safe + r_rate * torch.cos(phi) / cos_theta_safe

        # Dynamics (Position derivatives - use state angles)
        x_dot = (torch.cos(theta) * torch.cos(psi)) * u_vel + (torch.sin(phi) * torch.sin(theta) * torch.cos(psi) - torch.sin(psi) * torch.cos(phi)) * v_vel + \
                (torch.cos(phi) * torch.sin(theta) * torch.cos(psi) + torch.sin(phi) * torch.sin(psi)) * w_vel
        y_dot = (torch.cos(theta) * torch.sin(psi)) * u_vel + (torch.sin(phi) * torch.sin(theta) * torch.sin(psi) + torch.cos(psi) * torch.cos(phi)) * v_vel + \
                (torch.cos(phi) * torch.sin(theta) * torch.sin(psi) - torch.sin(phi) * torch.cos(psi)) * w_vel
        z_dot = (-torch.sin(theta)) * u_vel + (torch.sin(phi) * torch.cos(theta)) * v_vel + (torch.cos(phi) * torch.cos(theta)) * w_vel

        # Assume zero external disturbances if not provided
        zeros_like_input = torch.zeros_like(x_pos) # Create zeros on the correct device
        Fx_dist = zeros_like_input
        Fy_dist = zeros_like_input
        Fz_dist = zeros_like_input

        # Dynamics (Velocity derivatives - use state angles and rates)
        u_dot = r_rate * v_vel - q_rate * w_vel + self.g * torch.sin(theta) + Fx_dist
        v_dot = p_rate * w_vel - r_rate * u_vel - self.g * torch.sin(phi) * torch.cos(theta) + Fy_dist
        w_dot = q_rate * u_vel - p_rate * v_vel - self.g * torch.cos(phi) * torch.cos(theta) + (1 / self.mass) * (Fz_cmd) + Fz_dist

        # Angular accelerations (p_dot, q_dot, r_dot) are needed for a full 12-state derivative.
        # Since they are not defined, we will return only the first 9 derivatives.
        # Placeholder: If angular accelerations were calculated (e.g., from inertia and torques):
        # p_dot = ...
        # q_dot = ...
        # r_dot = ...

        # Return the state derivatives (first 9)
        derivatives = [
            x_dot,    # 0
            y_dot,    # 1
            z_dot,    # 2
            phi_dot,  # 3
            theta_dot,# 4
            psi_dot,  # 5
            u_dot,    # 6
            v_dot,    # 7
            w_dot     # 8
            # p_dot,  # 9 (Missing)
            # q_dot,  # 10 (Missing)
            # r_dot   # 11 (Missing)
        ]

        # Stack along the feature dimension (dim=-1) to maintain batch structure
        return torch.stack(derivatives, dim=-1) # Shape: [batch_size, 9]
