#!/usr/bin/env python3
"""
Data Logger for Training Metrics
"""
import csv
import json
import os
from datetime import datetime


class DataLogger:
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = os.path.join(log_dir, f'training_{timestamp}.csv')
        self.json_file = os.path.join(log_dir, f'summary_{timestamp}.json')
        
        self._initialize_csv()
        self.summary_data = {
            'start_time': timestamp,
            'total_episodes': 0,
            'best_reward': float('-inf'),
            'best_episode': 0
        }
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Total_Reward', 'Steps', 'Epsilon', 
                'Goal_Reached', 'Timestamp'
            ])
    
    def log_episode(self, episode, reward, steps, epsilon, goal_reached):
        """Log data from a training episode"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, steps, epsilon, goal_reached, timestamp
            ])
        
        # Update summary
        self.summary_data['total_episodes'] = episode + 1
        if reward > self.summary_data['best_reward']:
            self.summary_data['best_reward'] = reward
            self.summary_data['best_episode'] = episode
    
    def log_hyperparameters(self, params):
        """Log hyperparameters used for training"""
        self.summary_data['hyperparameters'] = params
    
    def save_summary(self):
        """Save summary data to JSON"""
        self.summary_data['end_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(self.json_file, 'w') as f:
            json.dump(self.summary_data, f, indent=4)
        
        print(f'Summary saved to {self.json_file}')
    
    def print_statistics(self, window=100):
        """Print training statistics"""
        print("\n" + "="*50)
        print(f"Training Statistics")
        print("="*50)
        print(f"Total Episodes: {self.summary_data['total_episodes']}")
        print(f"Best Reward: {self.summary_data['best_reward']:.2f}")
        print(f"Best Episode: {self.summary_data['best_episode']}")
        print("="*50 + "\n")
