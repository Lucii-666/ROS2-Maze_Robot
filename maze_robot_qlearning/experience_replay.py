#!/usr/bin/env python3
"""
Experience Replay Buffer for Q-Learning
"""
import random
from collections import deque
import pickle


class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        return random.sample(self.buffer, batch_size)
    
    def sample_recent(self, batch_size):
        """Sample from recent experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        recent = list(self.buffer)[-batch_size*2:]
        return random.sample(recent, min(batch_size, len(recent)))
    
    def get_prioritized_sample(self, batch_size, priority_func=None):
        """Sample with priority based on custom function"""
        if priority_func is None:
            return self.sample(batch_size)
        
        # Calculate priorities
        priorities = [priority_func(exp) for exp in self.buffer]
        total_priority = sum(priorities)
        
        if total_priority == 0:
            return self.sample(batch_size)
        
        # Normalize priorities
        probabilities = [p / total_priority for p in priorities]
        
        # Sample based on probabilities
        indices = random.choices(range(len(self.buffer)), 
                                weights=probabilities, 
                                k=min(batch_size, len(self.buffer)))
        
        return [self.buffer[i] for i in indices]
    
    def size(self):
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_full(self):
        """Check if buffer is full"""
        return len(self.buffer) >= self.capacity
    
    def clear(self):
        """Clear all experiences"""
        self.buffer.clear()
    
    def get_statistics(self):
        """Get buffer statistics"""
        if not self.buffer:
            return {
                'size': 0,
                'capacity': self.capacity,
                'avg_reward': 0,
                'positive_rewards': 0,
                'negative_rewards': 0
            }
        
        rewards = [exp[2] for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'avg_reward': sum(rewards) / len(rewards),
            'positive_rewards': sum(1 for r in rewards if r > 0),
            'negative_rewards': sum(1 for r in rewards if r < 0),
            'max_reward': max(rewards),
            'min_reward': min(rewards)
        }
    
    def save(self, filename):
        """Save buffer to file"""
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f'Experience buffer saved to {filename}')
    
    def load(self, filename):
        """Load buffer from file"""
        with open(filename, 'rb') as f:
            experiences = pickle.load(f)
        
        self.buffer.clear()
        for exp in experiences:
            self.buffer.append(exp)
        
        print(f'Experience buffer loaded from {filename}')
    
    def get_successful_experiences(self):
        """Get experiences that led to success"""
        successful = []
        for exp in self.buffer:
            state, action, reward, next_state, done = exp
            if reward > 50:  # High reward indicates success
                successful.append(exp)
        return successful
    
    def replay_to_agent(self, agent, batch_size=32):
        """Replay experiences to update agent"""
        if len(self.buffer) < batch_size:
            return
        
        batch = self.sample(batch_size)
        
        for state, action, reward, next_state, done in batch:
            agent.update_q_value(state, action, reward, next_state)
