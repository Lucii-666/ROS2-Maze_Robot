#!/usr/bin/env python3
"""
Training Script for Q-Learning Agent
"""
import rclpy
from rclpy.node import Node
from maze_robot_qlearning.q_learning_agent import QLearningAgent
from maze_robot_qlearning.maze_environment import MazeEnvironment
from maze_robot_qlearning.reward_calculator import RewardCalculator
import pickle
import time


class TrainingManager(Node):
    def __init__(self):
        super().__init__('training_manager')
        
        # Training parameters
        self.num_episodes = 1000
        self.max_steps_per_episode = 200
        self.current_episode = 0
        
        # Components
        self.agent = QLearningAgent()
        self.environment = MazeEnvironment()
        self.reward_calculator = RewardCalculator()
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        
        self.get_logger().info('Training Manager initialized')
    
    def train(self):
        """Main training loop"""
        for episode in range(self.num_episodes):
            self.current_episode = episode
            episode_reward = 0
            steps = 0
            
            # Reset environment
            state = self.environment.get_state()
            
            for step in range(self.max_steps_per_episode):
                # Choose and execute action
                action = self.agent.choose_action(state)
                self.environment.execute_action(action)
                time.sleep(0.1)  # Wait for action to execute
                
                # Get next state and reward
                next_state = self.environment.get_state()
                goal_reached = self.environment.is_goal_reached()
                
                reward = self.reward_calculator.calculate_reward(
                    state, next_state, self.environment.goal_position
                )
                
                # Update Q-table
                self.agent.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if goal_reached:
                    self.get_logger().info(f'Goal reached in episode {episode}!')
                    break
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            
            if episode % 10 == 0:
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                self.get_logger().info(
                    f'Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}'
                )
        
        self.save_model()
    
    def save_model(self):
        """Save Q-table to file"""
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.agent.q_table, f)
        self.get_logger().info('Model saved successfully')


def main(args=None):
    rclpy.init(args=args)
    trainer = TrainingManager()
    trainer.train()
    trainer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
