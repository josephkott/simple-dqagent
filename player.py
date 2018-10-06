#!/usr/local/bin/python3
import random
import numpy
import time
import os

# Using keras as API for TensorFlow neural network lib
from keras import Sequential
from keras.layers import Dense

from game import Game
from actions import POSSIBLE_ACTIONS, ACTION_TO_INDEX
from experience_replay import ExperienceReplay


class Player:
    """
    This class represents a player, his strategy of learning and playing the game.
    """
    def __init__(self):
        # Learning rate 
        self.alpha = 0.1

        # gamma is a parameter of Q - learing algorithm
        self.gamma = 0.9

        # We use epsilon - greedy strategy of learning
        # TODO: use epsilon decay strategy as well
        self.epsilon = 0.1
        
        # Number of epochs (fully played games) to study an agent
        self.epochs = 50000

        # Game to play
        self.game = Game()

        # Number of hidden layer nodes
        self.hidden_layer_nodes = 24

        # Create keras model
        # TODO: depict structure here
        self.model = Sequential()
        self.model.add(Dense(self.hidden_layer_nodes, input_dim=self.game.state_size, activation='relu'))
        self.model.add(Dense(self.hidden_layer_nodes, activation='relu'))
        self.model.add(Dense(len(POSSIBLE_ACTIONS), activation='linear'))

        # Initialize experience replay
        self.experience_replay = ExperienceReplay(size=100)
        self.batch_size = 10
    
    def train_model_on_batch(self):
        batch = self.experience_replay.get_batch(self.batch_size)

        for state, action, reward, next_state in batch:
            if self.game.is_over():
                target = reward
            else:
                target = reward + self.gamma * numpy.argmax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)[0]
            target_f[action] = target

            # TODO: it should be optimized ...
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def train(self, interactive=False):
        for _ in range(self.epochs):
            self.game.create_agent()

            while not self.game.is_over():
                if interactive:
                    os.system('clear')
                    self.game.show()
                    time.sleep(0.1)

                state = numpy.array(self.game.encode())
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(POSSIBLE_ACTIONS)
                else:
                    index = numpy.argmax(self.model.predict(state)[0])
                    action = POSSIBLE_ACTIONS[index]
                
                reward = self.game.act(action)
                next_state = numpy.array(self.game.encode())

                self.experience_replay.remember(state, action, reward, next_state)
                self.train_model_on_batch()

        print("Training finished!\n")
    
    def play(self, interactive=False):
        for _ in range(self.epochs):
            self.game.create_agent()

            while not self.game.is_over():
                if interactive:
                    os.system('clear')
                    self.game.show()
                    time.sleep(0.1)

                state = numpy.array(self.game.encode())
                
                # TODO: predict with NN
                action = None

                self.game.act(action)


# Run `python3 player.py` to learn agent and see how it plays
if __name__ == '__main__':
    player = Player()
    player.train(interactive=False)
    player.play(interactive=True)