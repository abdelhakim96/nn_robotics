import numpy as np
import math

# Added clockwise parameter to function definition
def generate_circular_trajectory(t, radius=2.0, speed=0.5, center_x=0, center_y=0, z_func=lambda t: 1.0, clockwise=False):
    """
    Generates points on a circular trajectory in the XY plane.

    Args:
        t (float): Current time.
        radius (float): Radius of the circle.
        speed (float): Tangential speed along the circle.
        center_x (float): X-coordinate of the circle center.
        center_y (float): Y-coordinate of the circle center.
        z_func (callable): Function defining Z coordinate based on time t.
        clockwise (bool): If True, generates clockwise motion.

    Returns:
        np.ndarray: Target position [x_ref, y_ref, z_ref].
    """
    if radius == 0: # Avoid division by zero for hover
        angle = 0
    else:
        angle = speed * t / radius
    
    direction = -1.0 if clockwise else 1.0
    x_ref = center_x + radius * np.cos(angle)
    y_ref = center_y + radius * direction * np.sin(angle)
    z_ref = z_func(t)
    return np.array([x_ref, y_ref, z_ref])

def generate_lemniscate_trajectory(t, scale=2.0, speed=0.3, center_x=0, center_y=0, z=1.0):
    """
    Generates points on a lemniscate (figure-eight) trajectory in the XY plane.
    Uses parametric equation: x = a*cos(t)/(1+sin(t)^2), y = a*sin(t)cos(t)/(1+sin(t)^2)

    Args:
        t (float): Current time (acts as parameter, speed scales it).
        scale (float): Size parameter 'a' of the lemniscate.
        speed (float): Factor controlling speed along the curve.
        center_x (float): X-coordinate of the center.
        center_y (float): Y-coordinate of the center.
        z (float): Constant Z coordinate.

    Returns:
        np.ndarray: Target position [x_ref, y_ref, z_ref].
    """
    angle = speed * t
    denominator = (1 + np.sin(angle)**2)
    if abs(denominator) < 1e-6: # Avoid division by zero
        denominator = 1e-6
    x_ref = center_x + scale * np.cos(angle) / denominator
    y_ref = center_y + scale * np.sin(angle) * np.cos(angle) / denominator
    z_ref = z
    return np.array([x_ref, y_ref, z_ref])

def generate_sinusoidal_hover(t, axis='z', amplitude=0.5, frequency=0.5, hover_pos=[0,0,1]):
    """
    Generates a hover trajectory with sinusoidal motion along one axis.

    Args:
        t (float): Current time.
        axis (str): Axis of sinusoidal motion ('x', 'y', or 'z').
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave (Hz).
        hover_pos (list): Base hover position [x, y, z].

    Returns:
        np.ndarray: Target position [x_ref, y_ref, z_ref].
    """
    pos = np.array(hover_pos, dtype=np.float32)
    offset = amplitude * np.sin(2 * np.pi * frequency * t)
    if axis == 'x':
        pos[0] += offset
    elif axis == 'y':
        pos[1] += offset
    else: # Default to z
        pos[2] += offset
    return pos

def generate_helix_trajectory(t, radius=1.5, speed=0.4, pitch=0.5, center_x=0, center_y=0, z_start=0.1, clockwise=False):
    """
    Generates points on a helical trajectory.

    Args:
        t (float): Current time.
        radius (float): Radius of the helix cylinder.
        speed (float): Speed along the helical path (approximate).
        pitch (float): Vertical distance covered per revolution.
        center_x (float): X-coordinate of the helix center.
        center_y (float): Y-coordinate of the helix center.
        z_start (float): Starting Z coordinate at t=0.
        clockwise (bool): If True, generates clockwise rotation when viewed from above.

    Returns:
        np.ndarray: Target position [x_ref, y_ref, z_ref].
    """
    if radius == 0: # Avoid division by zero
        angle = 0
        z_ref = z_start + speed * t # Treat as vertical line
    else:
        angular_speed = speed / radius # Approximate angular speed
        angle = angular_speed * t
        direction = -1.0 if clockwise else 1.0
        z_ref = z_start + (pitch / (2 * np.pi)) * angle * direction # z increases with angle

    x_ref = center_x + radius * np.cos(angle)
    y_ref = center_y + radius * direction * np.sin(angle)
    return np.array([x_ref, y_ref, z_ref])

def generate_accelerating_line(t, start_pos=[0,0,1], direction=[1,0,0], initial_speed=0.1, acceleration=0.2):
    """
    Generates points on a line with constant acceleration.

    Args:
        t (float): Current time.
        start_pos (list): Starting position [x, y, z].
        direction (list): Unit vector indicating direction of motion.
        initial_speed (float): Speed at t=0.
        acceleration (float): Constant acceleration along the direction vector.

    Returns:
        np.ndarray: Target position [x_ref, y_ref, z_ref].
    """
    start_pos = np.array(start_pos, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)
    # Normalize direction vector
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    else:
        direction = np.array([1.0, 0, 0]) # Default direction if zero vector given

    distance = initial_speed * t + 0.5 * acceleration * t**2
    pos = start_pos + direction * distance
    return pos
