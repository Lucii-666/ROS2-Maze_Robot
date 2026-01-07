#!/usr/bin/env python3
"""
Action Space Definition for Robot Navigation
"""


class ActionSpace:
    def __init__(self):
        # Define available actions
        self.actions = {
            0: 'forward',
            1: 'backward',
            2: 'left',
            3: 'right',
            4: 'forward_left',
            5: 'forward_right',
            6: 'stop'
        }
        
        # Action parameters
        self.action_params = {
            'forward': {'linear_x': 0.5, 'angular_z': 0.0},
            'backward': {'linear_x': -0.3, 'angular_z': 0.0},
            'left': {'linear_x': 0.0, 'angular_z': 0.5},
            'right': {'linear_x': 0.0, 'angular_z': -0.5},
            'forward_left': {'linear_x': 0.4, 'angular_z': 0.3},
            'forward_right': {'linear_x': 0.4, 'angular_z': -0.3},
            'stop': {'linear_x': 0.0, 'angular_z': 0.0}
        }
    
    def get_action_name(self, action_id):
        """Get action name from ID"""
        return self.actions.get(action_id, 'unknown')
    
    def get_action_params(self, action_id):
        """Get velocity parameters for action"""
        action_name = self.get_action_name(action_id)
        return self.action_params.get(action_name, {'linear_x': 0.0, 'angular_z': 0.0})
    
    def get_num_actions(self):
        """Get total number of actions"""
        return len(self.actions)
    
    def get_valid_actions(self, state=None, obstacles=None):
        """Get valid actions for current state"""
        # Basic implementation - can be extended with obstacle checking
        return list(self.actions.keys())
    
    def get_opposite_action(self, action_id):
        """Get opposite action"""
        opposites = {
            0: 1,  # forward <-> backward
            1: 0,
            2: 3,  # left <-> right
            3: 2,
            4: 5,  # forward_left <-> forward_right
            5: 4,
            6: 6   # stop <-> stop
        }
        return opposites.get(action_id, action_id)
    
    def action_to_grid_change(self, action_id):
        """Convert action to grid position change"""
        # Simplified grid-based movement
        changes = {
            0: (1, 0),   # forward
            1: (-1, 0),  # backward
            2: (0, 1),   # left
            3: (0, -1),  # right
            4: (1, 1),   # forward_left
            5: (1, -1),  # forward_right
            6: (0, 0)    # stop
        }
        return changes.get(action_id, (0, 0))
    
    def set_action_speeds(self, linear_speed=None, angular_speed=None):
        """Update action speed parameters"""
        if linear_speed is not None:
            for action in ['forward', 'forward_left', 'forward_right']:
                self.action_params[action]['linear_x'] = linear_speed
        
        if angular_speed is not None:
            self.action_params['left']['angular_z'] = angular_speed
            self.action_params['right']['angular_z'] = -angular_speed
