#!/usr/bin/env python3
"""
Performance Metrics Calculator
"""
import numpy as np


class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'total_steps': 0,
            'total_rewards': 0,
            'collision_count': 0
        }
        
        self.episode_data = []
    
    def record_episode(self, steps, reward, success, collision):
        """Record data from a completed episode"""
        self.metrics['total_episodes'] += 1
        self.metrics['total_steps'] += steps
        self.metrics['total_rewards'] += reward
        
        if success:
            self.metrics['successful_episodes'] += 1
        else:
            self.metrics['failed_episodes'] += 1
        
        if collision:
            self.metrics['collision_count'] += 1
        
        self.episode_data.append({
            'steps': steps,
            'reward': reward,
            'success': success,
            'collision': collision
        })
    
    def get_success_rate(self):
        """Calculate overall success rate"""
        if self.metrics['total_episodes'] == 0:
            return 0.0
        return self.metrics['successful_episodes'] / self.metrics['total_episodes']
    
    def get_average_steps(self):
        """Calculate average steps per episode"""
        if self.metrics['total_episodes'] == 0:
            return 0.0
        return self.metrics['total_steps'] / self.metrics['total_episodes']
    
    def get_average_reward(self):
        """Calculate average reward per episode"""
        if self.metrics['total_episodes'] == 0:
            return 0.0
        return self.metrics['total_rewards'] / self.metrics['total_episodes']
    
    def get_collision_rate(self):
        """Calculate collision rate"""
        if self.metrics['total_episodes'] == 0:
            return 0.0
        return self.metrics['collision_count'] / self.metrics['total_episodes']
    
    def get_efficiency_score(self):
        """Calculate efficiency score (lower steps with higher success is better)"""
        success_rate = self.get_success_rate()
        avg_steps = self.get_average_steps()
        
        if avg_steps == 0:
            return 0.0
        
        # Higher success rate and lower steps = better efficiency
        efficiency = success_rate / (avg_steps / 100)
        return efficiency
    
    def get_learning_curve(self, window=50):
        """Get smoothed learning curve data"""
        if len(self.episode_data) < window:
            return []
        
        rewards = [ep['reward'] for ep in self.episode_data]
        smoothed = []
        
        for i in range(len(rewards) - window + 1):
            window_avg = np.mean(rewards[i:i+window])
            smoothed.append(window_avg)
        
        return smoothed
    
    def get_improvement_rate(self, early_window=100, late_window=100):
        """Calculate improvement from early to late training"""
        total_eps = len(self.episode_data)
        
        if total_eps < early_window + late_window:
            return 0.0
        
        early_rewards = [ep['reward'] for ep in self.episode_data[:early_window]]
        late_rewards = [ep['reward'] for ep in self.episode_data[-late_window:]]
        
        early_avg = np.mean(early_rewards)
        late_avg = np.mean(late_rewards)
        
        if early_avg == 0:
            return 0.0
        
        improvement = ((late_avg - early_avg) / abs(early_avg)) * 100
        return improvement
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Episodes: {self.metrics['total_episodes']}")
        print(f"Successful Episodes: {self.metrics['successful_episodes']}")
        print(f"Success Rate: {self.get_success_rate()*100:.2f}%")
        print(f"\nAverage Steps per Episode: {self.get_average_steps():.2f}")
        print(f"Average Reward per Episode: {self.get_average_reward():.2f}")
        print(f"Collision Rate: {self.get_collision_rate()*100:.2f}%")
        print(f"\nEfficiency Score: {self.get_efficiency_score():.4f}")
        
        improvement = self.get_improvement_rate()
        print(f"Improvement Rate: {improvement:+.2f}%")
        print("="*60 + "\n")
    
    def export_metrics(self):
        """Export metrics as dictionary"""
        return {
            'summary': self.metrics,
            'success_rate': self.get_success_rate(),
            'average_steps': self.get_average_steps(),
            'average_reward': self.get_average_reward(),
            'collision_rate': self.get_collision_rate(),
            'efficiency_score': self.get_efficiency_score(),
            'improvement_rate': self.get_improvement_rate()
        }
