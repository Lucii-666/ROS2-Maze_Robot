#!/usr/bin/env python3
"""
Visualization Tool for Training Progress
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle


class TrainingVisualizer:
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_values = []
    
    def add_episode_data(self, reward, steps, epsilon):
        """Add data from a training episode"""
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.epsilon_values.append(epsilon)
    
    def plot_training_progress(self, save_path='training_progress.png'):
        """Plot training metrics"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rewards
        axes[0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        axes[0].plot(self._moving_average(self.episode_rewards, 50), 
                     linewidth=2, label='Moving Average (50)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Rewards Over Time')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot steps
        axes[1].plot(self.episode_steps, alpha=0.6, label='Steps per Episode')
        axes[1].plot(self._moving_average(self.episode_steps, 50), 
                     linewidth=2, label='Moving Average (50)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Steps per Episode')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot epsilon
        axes[2].plot(self.epsilon_values, linewidth=2)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Epsilon')
        axes[2].set_title('Exploration Rate (Epsilon) Decay')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f'Training progress saved to {save_path}')
    
    def plot_q_table_heatmap(self, q_table, save_path='q_table_heatmap.png'):
        """Visualize Q-table as heatmap"""
        grid_size = 10
        value_map = np.zeros((grid_size, grid_size))
        
        for (state, action), value in q_table.items():
            if isinstance(state, tuple) and len(state) == 2:
                x, y = state
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    value_map[y, x] = max(value_map[y, x], value)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(value_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Max Q-Value')
        plt.title('Q-Table Heatmap (Max Q-Value per State)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f'Q-table heatmap saved to {save_path}')
    
    def _moving_average(self, data, window_size):
        """Calculate moving average"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def save_data(self, filename='training_data.pkl'):
        """Save training data"""
        data = {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'epsilon': self.epsilon_values
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'Training data saved to {filename}')
    
    def load_data(self, filename='training_data.pkl'):
        """Load training data"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.episode_rewards = data['rewards']
        self.episode_steps = data['steps']
        self.epsilon_values = data['epsilon']
        print(f'Training data loaded from {filename}')


if __name__ == '__main__':
    # Example usage
    visualizer = TrainingVisualizer()
    visualizer.load_data()
    visualizer.plot_training_progress()
