#!/usr/local/bin/python3
import random


class ExperienceReplay:
    def __init__(self, size):
        self.size = size
        self.memory = []
    
    def remember(self, state, action, reward, next_state, is_over):
        self.memory.append((state, action, reward, next_state, is_over))
        if len(self.memory) > self.size:
            del self.memory[0]
    
    def get_batch(self, batch_size):
        if len(self.memory) < batch_size:
            return self.memory
        return random.sample(self.memory, batch_size)
