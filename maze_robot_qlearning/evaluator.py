#!/usr/bin/env python3
"""
Evaluation Script for Trained Q-Learning Agent
"""
import rclpy
from rclpy.node import Node
import pickle
import time


class AgentEvaluator(Node):
    def __init__(self, model_path='q_table.pkl'):
        super().__init__('agent_evaluator')
        
        self.model_path = model_path
        self.q_table = None
        self.load_model()
        
        self.evaluation_episodes = 100
        self.results = {
            'success_count': 0,
            'total_steps': [],
            'total_rewards': [],
            'completion_times': []
        }
        
        self.get_logger().info('Agent Evaluator initialized')
    
    def load_model(self):
        """Load trained Q-table"""
        try:
            with open(self.model_path, 'rb') as f:
                self.q_table = pickle.load(f)
            self.get_logger().info(f'Model loaded from {self.model_path}')
            self.get_logger().info(f'Q-table size: {len(self.q_table)} entries')
        except FileNotFoundError:
            self.get_logger().error(f'Model file not found: {self.model_path}')
            self.q_table = {}
    
    def get_best_action(self, state, num_actions=4):
        """Get best action from Q-table (no exploration)"""
        q_values = []
        for action in range(num_actions):
            q_values.append(self.q_table.get((state, action), 0.0))
        
        best_action = q_values.index(max(q_values))
        return best_action
    
    def evaluate_episode(self, environment, max_steps=200):
        """Evaluate agent for one episode"""
        state = environment.get_state()
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        for step in range(max_steps):
            # Get best action (greedy policy)
            action = self.get_best_action(state)
            
            # Execute action
            environment.execute_action(action)
            time.sleep(0.1)
            
            # Get next state
            next_state = environment.get_state()
            
            # Check if goal reached
            if environment.is_goal_reached():
                success = True
                total_reward += 100
                break
            
            steps += 1
            state = next_state
        else:
            success = False
        
        completion_time = time.time() - start_time
        
        return {
            'success': success,
            'steps': steps,
            'reward': total_reward,
            'time': completion_time
        }
    
    def run_evaluation(self, environment):
        """Run complete evaluation"""
        self.get_logger().info(f'Starting evaluation for {self.evaluation_episodes} episodes')
        
        for episode in range(self.evaluation_episodes):
            result = self.evaluate_episode(environment)
            
            if result['success']:
                self.results['success_count'] += 1
            
            self.results['total_steps'].append(result['steps'])
            self.results['total_rewards'].append(result['reward'])
            self.results['completion_times'].append(result['time'])
            
            if (episode + 1) % 10 == 0:
                self.get_logger().info(f'Evaluated {episode + 1}/{self.evaluation_episodes} episodes')
        
        self.print_evaluation_results()
    
    def print_evaluation_results(self):
        """Print evaluation statistics"""
        success_rate = self.results['success_count'] / self.evaluation_episodes
        avg_steps = sum(self.results['total_steps']) / len(self.results['total_steps'])
        avg_reward = sum(self.results['total_rewards']) / len(self.results['total_rewards'])
        avg_time = sum(self.results['completion_times']) / len(self.results['completion_times'])
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Episodes: {self.evaluation_episodes}")
        print(f"Successful Episodes: {self.results['success_count']}")
        print(f"Success Rate: {success_rate*100:.2f}%")
        print(f"\nAverage Steps: {avg_steps:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Completion Time: {avg_time:.2f}s")
        print("="*60 + "\n")
    
    def save_evaluation_results(self, filename='evaluation_results.pkl'):
        """Save evaluation results"""
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        self.get_logger().info(f'Evaluation results saved to {filename}')


def main(args=None):
    rclpy.init(args=args)
    evaluator = AgentEvaluator()
    # Note: Need to create environment instance to run evaluation
    # evaluator.run_evaluation(environment)
    rclpy.spin(evaluator)
    evaluator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
