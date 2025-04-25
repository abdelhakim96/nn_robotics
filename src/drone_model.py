import torch
import torch.nn as nn

class DroneModel(nn.Module):
    def __init__(self, mass, inertia):
        super().__init__()
        self.mass = mass
        self.inertia = inertia

    def forward(self, state, control_input):
        # Basic drone dynamics (placeholder)
        x, y, z, roll, pitch, yaw, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot = state
        thrust = control_input[0]
        torque_roll = control_input[1]
        torque_pitch = control_input[2]
        torque_yaw = control_input[3]

        x_acc = 0  # Placeholder
        y_acc = 0  # Placeholder
        z_acc = (thrust / self.mass) - 9.81  # Placeholder
        roll_acc = torque_roll / self.inertia[0]  # Placeholder
        pitch_acc = torque_pitch / self.inertia[1]  # Placeholder
        yaw_acc = torque_yaw / self.inertia[2]  # Placeholder

        next_state = torch.tensor([
            x + x_dot,
            y + y_dot,
            z + z_dot,
            roll + roll_dot,
            pitch + pitch_dot,
            yaw + yaw_dot,
            x_dot + x_acc,
            y_dot + y_acc,
            z_dot + z_acc,
            roll_dot + roll_acc,
            pitch_dot + pitch_acc,
            yaw_dot + yaw_acc
        ])

        return next_state
