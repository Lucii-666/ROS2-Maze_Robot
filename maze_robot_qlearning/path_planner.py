#!/usr/bin/env python3
"""
Path Planner using A* Algorithm
"""
import heapq
import numpy as np


class PathPlanner:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.obstacle_map = np.zeros((grid_size, grid_size), dtype=bool)
    
    def set_obstacles(self, obstacles):
        """Set obstacle positions on grid"""
        self.obstacle_map.fill(False)
        for obs in obstacles:
            x, y = obs
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.obstacle_map[y, x] = True
    
    def is_valid(self, position):
        """Check if position is valid and not an obstacle"""
        x, y = position
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        return not self.obstacle_map[y, x]
    
    def get_neighbors(self, position):
        """Get valid neighboring positions"""
        x, y = position
        neighbors = []
        
        # 4-connected movement
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def heuristic(self, pos1, pos2):
        """Calculate Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def a_star(self, start, goal):
        """Find shortest path using A* algorithm"""
        if not self.is_valid(start) or not self.is_valid(goal):
            return None
        
        # Priority queue: (f_score, position)
        open_set = [(0, start)]
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from start to goal"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def get_next_action(self, current_pos, goal_pos):
        """Get next action to move towards goal"""
        path = self.a_star(current_pos, goal_pos)
        
        if path is None or len(path) < 2:
            return None
        
        next_pos = path[1]
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        # Convert direction to action
        if dx == 1:
            return 0  # forward
        elif dx == -1:
            return 1  # backward
        elif dy == 1:
            return 2  # left
        elif dy == -1:
            return 3  # right
        
        return None
    
    def visualize_path(self, path):
        """Visualize path on grid"""
        grid = self.obstacle_map.astype(int) * 2
        
        if path:
            for pos in path:
                grid[pos[1], pos[0]] = 1
            
            # Mark start and end
            grid[path[0][1], path[0][0]] = 3
            grid[path[-1][1], path[-1][0]] = 4
        
        symbols = {0: '□', 1: '·', 2: '■', 3: 'S', 4: 'G'}
        
        for row in grid:
            print(' '.join(symbols[cell] for cell in row))
