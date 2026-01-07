#!/usr/bin/env python3
"""
Hyperparameter Tuning Utilities
"""
import numpy as np
import itertools
import json
from datetime import datetime


class HyperparameterTuner:
    def __init__(self):
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')
    
    def grid_search(self, param_grid, eval_function, episodes=100):
        """Perform grid search over parameter space"""
        print("Starting Grid Search...")
        print(f"Parameter grid: {param_grid}")
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        total_combinations = len(combinations)
        print(f"Total combinations to test: {total_combinations}\n")
        
        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"Testing {idx+1}/{total_combinations}: {params}")
            
            # Evaluate parameters
            score = eval_function(params, episodes)
            
            result = {
                'params': params,
                'score': score,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.results.append(result)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"  New best score: {score:.4f}")
            else:
                print(f"  Score: {score:.4f}")
            print()
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def random_search(self, param_ranges, eval_function, n_iterations=50, episodes=100):
        """Perform random search over parameter space"""
        print("Starting Random Search...")
        print(f"Iterations: {n_iterations}\n")
        
        for iteration in range(n_iterations):
            # Sample random parameters
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Continuous range
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Discrete choices
                    params[param_name] = np.random.choice(param_range)
            
            print(f"Iteration {iteration+1}/{n_iterations}: {params}")
            
            # Evaluate parameters
            score = eval_function(params, episodes)
            
            result = {
                'params': params,
                'score': score,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.results.append(result)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"  New best score: {score:.4f}")
            else:
                print(f"  Score: {score:.4f}")
            print()
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def save_results(self, filename='tuning_results.json'):
        """Save tuning results to file"""
        data = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Results saved to {filename}")
    
    def load_results(self, filename='tuning_results.json'):
        """Load tuning results from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.best_params = data['best_params']
        self.best_score = data['best_score']
        self.results = data['all_results']
        
        print(f"Results loaded from {filename}")
    
    def get_top_n_params(self, n=5):
        """Get top N parameter combinations"""
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:n]
    
    def plot_parameter_importance(self):
        """Analyze which parameters have most impact"""
        if not self.results:
            print("No results to analyze")
            return
        
        param_names = list(self.results[0]['params'].keys())
        importance = {param: 0 for param in param_names}
        
        # Simple variance-based importance
        for param in param_names:
            values = [r['params'][param] for r in self.results]
            scores = [r['score'] for r in self.results]
            
            # Calculate correlation between parameter and score
            if len(set(values)) > 1:
                importance[param] = abs(np.corrcoef(values, scores)[0, 1])
        
        print("\nParameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.4f}")


def example_eval_function(params, episodes):
    """Example evaluation function for testing"""
    # Simulate training and return a score
    learning_rate = params.get('learning_rate', 0.1)
    discount_factor = params.get('discount_factor', 0.99)
    epsilon_decay = params.get('epsilon_decay', 0.995)
    
    # Simple scoring function
    score = learning_rate * 10 + discount_factor * 5 + epsilon_decay * 3
    return score


if __name__ == '__main__':
    # Example usage
    tuner = HyperparameterTuner()
    
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.5],
        'discount_factor': [0.9, 0.99],
        'epsilon_decay': [0.99, 0.995, 0.999]
    }
    
    best_params, best_score = tuner.grid_search(param_grid, example_eval_function)
