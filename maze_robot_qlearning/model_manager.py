#!/usr/bin/env python3
"""
Model Manager for Saving and Loading Q-Tables
"""
import pickle
import os
import json
from datetime import datetime


class ModelManager:
    def __init__(self, model_dir='./models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, q_table, metadata=None, model_name=None):
        """Save Q-table and metadata"""
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f'q_table_{timestamp}'
        
        # Save Q-table
        q_table_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        with open(q_table_path, 'wb') as f:
            pickle.dump(q_table, f)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata['save_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata['num_states'] = len(q_table)
        
        metadata_path = os.path.join(self.model_dir, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f'Model saved: {q_table_path}')
        print(f'Metadata saved: {metadata_path}')
        
        return q_table_path
    
    def load_model(self, model_name):
        """Load Q-table and metadata"""
        q_table_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        metadata_path = os.path.join(self.model_dir, f'{model_name}_metadata.json')
        
        # Load Q-table
        if not os.path.exists(q_table_path):
            raise FileNotFoundError(f'Model not found: {q_table_path}')
        
        with open(q_table_path, 'rb') as f:
            q_table = pickle.load(f)
        
        # Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f'Model loaded: {q_table_path}')
        print(f'States in Q-table: {len(q_table)}')
        
        return q_table, metadata
    
    def list_models(self):
        """List all saved models"""
        models = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl'):
                model_name = file[:-4]
                metadata_path = os.path.join(self.model_dir, f'{model_name}_metadata.json')
                
                info = {'name': model_name}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        info['metadata'] = json.load(f)
                
                models.append(info)
        
        return models
    
    def delete_model(self, model_name):
        """Delete a saved model"""
        q_table_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        metadata_path = os.path.join(self.model_dir, f'{model_name}_metadata.json')
        
        if os.path.exists(q_table_path):
            os.remove(q_table_path)
            print(f'Deleted: {q_table_path}')
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f'Deleted: {metadata_path}')
    
    def get_best_model(self):
        """Get the model with the best performance"""
        models = self.list_models()
        
        if not models:
            return None
        
        best_model = None
        best_reward = float('-inf')
        
        for model in models:
            if 'metadata' in model and 'best_reward' in model['metadata']:
                reward = model['metadata']['best_reward']
                if reward > best_reward:
                    best_reward = reward
                    best_model = model['name']
        
        return best_model
