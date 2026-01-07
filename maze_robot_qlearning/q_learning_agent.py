#!/usr/bin/env python3
"""
Q-Learning Agent for Maze Navigation
"""
import numpy as np
import rclpy
from rclpy.node import Node


class QLearningAgent(Node):
    def __init__(self):
        super().__init__('q_learning_agent')
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Initialize Q-table
        self.q_table = {}
        
        self.get_logger().info('Q-Learning Agent initialized')
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in range(4)], default=0.0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def choose_action(self, state, num_actions=4):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(num_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(num_actions)]
            return np.argmax(q_values)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def main(args=None):
    rclpy.init(args=args)
    agent = QLearningAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
