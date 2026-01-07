#!/usr/bin/env python3
"""
State Representation for Maze Navigation
"""
import numpy as np


class StateRepresentation:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.state_dim = 2  # (x, y) position
    
    def position_to_state(self, x, y, cell_size=1.0):
        """Convert continuous position to discrete grid state"""
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        
        # Clip to grid boundaries
        grid_x = max(0, min(grid_x, self.grid_size - 1))
        grid_y = max(0, min(grid_y, self.grid_size - 1))
        
        return (grid_x, grid_y)
    
    def state_to_position(self, state, cell_size=1.0):
        """Convert discrete state to continuous position (center of cell)"""
        x = (state[0] + 0.5) * cell_size
        y = (state[1] + 0.5) * cell_size
        return (x, y)
    
    def get_state_with_orientation(self, x, y, theta, cell_size=1.0, num_orientations=4):
        """Get state including orientation"""
        grid_x, grid_y = self.position_to_state(x, y, cell_size)
        
        # Discretize orientation
        orientation = int((theta / (2 * np.pi)) * num_orientations) % num_orientations
        
        return (grid_x, grid_y, orientation)
    
    def get_state_with_sensor(self, x, y, sensor_readings, cell_size=1.0):
        """Get state including sensor information"""
        grid_x, grid_y = self.position_to_state(x, y, cell_size)
        
        # Discretize sensor readings into zones
        front_clear = sensor_readings[0] > 0.5
        left_clear = sensor_readings[1] > 0.5
        right_clear = sensor_readings[2] > 0.5
        
        sensor_state = (front_clear, left_clear, right_clear)
        
        return (grid_x, grid_y, sensor_state)
    
    def get_neighbors(self, state):
        """Get valid neighboring states"""
        x, y = state
        neighbors = []
        
        # Four-connected neighbors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def get_distance(self, state1, state2, metric='manhattan'):
        """Calculate distance between two states"""
        if metric == 'manhattan':
            return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])
        elif metric == 'euclidean':
            return np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)
        elif metric == 'chebyshev':
            return max(abs(state1[0] - state2[0]), abs(state1[1] - state2[1]))
        else:
            raise ValueError(f'Unknown metric: {metric}')
    
    def is_valid_state(self, state):
        """Check if state is within grid boundaries"""
        x, y = state[:2]  # Handle states with additional dimensions
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def get_total_states(self):
        """Get total number of possible states"""
        return self.grid_size * self.grid_size
