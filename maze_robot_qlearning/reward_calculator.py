#!/usr/bin/env python3
"""
Reward Calculator for Maze Navigation
"""
import numpy as np


class RewardCalculator:
    def __init__(self):
        self.goal_reward = 100.0
        self.step_penalty = -1.0
        self.collision_penalty = -50.0
        self.distance_reward_scale = 10.0
    
    def calculate_reward(self, state, next_state, goal_state, collision=False):
        """Calculate reward based on state transition"""
        
        # Check if goal reached
        if next_state == goal_state:
            return self.goal_reward
        
        # Penalty for collision
        if collision:
            return self.collision_penalty
        
        # Distance-based reward
        current_distance = self._manhattan_distance(state, goal_state)
        next_distance = self._manhattan_distance(next_state, goal_state)
        distance_reward = (current_distance - next_distance) * self.distance_reward_scale
        
        # Step penalty to encourage efficiency
        total_reward = distance_reward + self.step_penalty
        
        return total_reward
    
    def _manhattan_distance(self, state1, state2):
        """Calculate Manhattan distance between two states"""
        return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])
    
    def _euclidean_distance(self, state1, state2):
        """Calculate Euclidean distance between two states"""
        return np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)
    
    def set_rewards(self, goal=None, step=None, collision=None, distance_scale=None):
        """Update reward values"""
        if goal is not None:
            self.goal_reward = goal
        if step is not None:
            self.step_penalty = step
        if collision is not None:
            self.collision_penalty = collision
        if distance_scale is not None:
            self.distance_reward_scale = distance_scale
