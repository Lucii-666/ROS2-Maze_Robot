#!/usr/bin/env python3
"""
Episode Manager for Training Control
"""
import time


class EpisodeManager:
    def __init__(self, max_episodes=1000, max_steps=200):
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps
        
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = None
        
        self.statistics = {
            'episode_rewards': [],
            'episode_steps': [],
            'episode_times': [],
            'success_count': 0,
            'collision_count': 0
        }
    
    def start_episode(self):
        """Start a new episode"""
        self.current_step = 0
        self.episode_start_time = time.time()
    
    def step(self):
        """Increment step counter"""
        self.current_step += 1
    
    def end_episode(self, total_reward, success=False, collision=False):
        """End current episode and record statistics"""
        episode_time = time.time() - self.episode_start_time
        
        self.statistics['episode_rewards'].append(total_reward)
        self.statistics['episode_steps'].append(self.current_step)
        self.statistics['episode_times'].append(episode_time)
        
        if success:
            self.statistics['success_count'] += 1
        if collision:
            self.statistics['collision_count'] += 1
        
        self.current_episode += 1
    
    def is_episode_done(self):
        """Check if episode should end"""
        return self.current_step >= self.max_steps_per_episode
    
    def is_training_done(self):
        """Check if training is complete"""
        return self.current_episode >= self.max_episodes
    
    def get_success_rate(self, window=100):
        """Calculate success rate over recent episodes"""
        if self.current_episode == 0:
            return 0.0
        
        recent_episodes = min(window, self.current_episode)
        recent_successes = self.statistics['success_count']
        
        # For more accurate calculation, need to track successes per episode
        return recent_successes / self.current_episode
    
    def get_average_reward(self, window=100):
        """Calculate average reward over recent episodes"""
        if not self.statistics['episode_rewards']:
            return 0.0
        
        recent_rewards = self.statistics['episode_rewards'][-window:]
        return sum(recent_rewards) / len(recent_rewards)
    
    def get_average_steps(self, window=100):
        """Calculate average steps over recent episodes"""
        if not self.statistics['episode_steps']:
            return 0
        
        recent_steps = self.statistics['episode_steps'][-window:]
        return sum(recent_steps) / len(recent_steps)
    
    def print_progress(self, window=10):
        """Print training progress"""
        if self.current_episode % window == 0 and self.current_episode > 0:
            avg_reward = self.get_average_reward(window)
            avg_steps = self.get_average_steps(window)
            success_rate = self.get_success_rate()
            
            print(f"\nEpisode {self.current_episode}/{self.max_episodes}")
            print(f"  Avg Reward (last {window}): {avg_reward:.2f}")
            print(f"  Avg Steps (last {window}): {avg_steps:.1f}")
            print(f"  Success Rate: {success_rate*100:.1f}%")
            print(f"  Collisions: {self.statistics['collision_count']}")
    
    def reset(self):
        """Reset episode manager"""
        self.current_episode = 0
        self.current_step = 0
        self.statistics = {
            'episode_rewards': [],
            'episode_steps': [],
            'episode_times': [],
            'success_count': 0,
            'collision_count': 0
        }
