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
        # gamma is a parameter of Q - learing algorithm
        self.gamma = 0.9

        # We use epsilon - greedy strategy of learning
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        
        # Number of epochs (fully played games) to study an agent
        self.epochs = 500

        # Game to play
        self.game = Game()

        # Number of hidden layer nodes
        self.hidden_layer_nodes = 20

        # Create keras model
        # _________________________________________________________________
        # Layer (type)                 Output Shape              Param #   
        # =================================================================
        # dense_1 (Dense)              (None, 20)                120       
        # _________________________________________________________________
        # dense_2 (Dense)              (None, 20)                420       
        # _________________________________________________________________
        # dense_3 (Dense)              (None, 5)                 105       
        # =================================================================
        # Total params: 645
        # Trainable params: 645
        # Non-trainable params: 0
        # _________________________________________________________________
        self.model = Sequential()
        self.model.add(Dense(self.hidden_layer_nodes, input_dim=self.game.state_size, activation='relu'))
        self.model.add(Dense(self.hidden_layer_nodes, activation='relu'))
        self.model.add(Dense(len(POSSIBLE_ACTIONS), activation='linear'))
        self.model.compile('Adam', loss='mse')

        # Initialize experience replay
        self.experience_replay = ExperienceReplay(size=2000)
        self.batch_size = 20
        self.max_turns = 100
    
    def train_model_on_batch(self):
        batch = self.experience_replay.get_batch(self.batch_size)

        # ---------------------------------- #
        # TODO: move this logic to get_batch
        states = []
        target_fs = []
        actions = []
        rewards = []
        next_states = []
        not_is_overs = []

        for state, action, reward, next_state, is_over in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            not_is_overs.append(not is_over)

        states = numpy.array(states)
        next_states = numpy.array(next_states)
        not_is_overs = numpy.array(not_is_overs)
        rewards = numpy.array(rewards)
        # ---------------------------------- #

        targets = rewards + not_is_overs * self.gamma * numpy.amax(self.model.predict(next_states), axis=1)
        target_fs = self.model.predict(states)

        for i in range(len(batch)):
            target_fs[i, ACTION_TO_INDEX[actions[i]]] = targets[i]
        self.model.fit(states, target_fs, verbose=0)

    def train(self, interactive=False):
        for epoch in range(self.epochs):
            self.game.create_agent()

            turns = 0
            while turns < self.max_turns:
                turns += 1

                if interactive:
                    os.system('clear')
                    self.game.show()
                    time.sleep(0.1)

                state = numpy.array(self.game.encode())
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(POSSIBLE_ACTIONS)
                else:
                    index = numpy.argmax(self.model.predict(state[numpy.newaxis])[0])
                    action = POSSIBLE_ACTIONS[index]
                
                reward = self.game.act(action)
                next_state = numpy.array(self.game.encode())

                is_over = self.game.is_over()
                if is_over:
                    reward -= 10
                    self.experience_replay.remember(state, action, reward, next_state, is_over)
                    break

                if turns == self.max_turns:
                    reward += 10

                self.experience_replay.remember(state, action, reward, next_state, is_over)
                self.train_model_on_batch()
            
            # Epsilon decay technic
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            print('Epoch: %i Total turns: %i' % (epoch, turns))

        print("Training finished!\n")
    
    def play(self, interactive=False):
        for _ in range(self.epochs):
            self.game.create_agent()

            while not self.game.is_over():
                if interactive:
                    os.system('clear')
                    self.game.show()
                    time.sleep(0.1)

                state = numpy.array(self.game.encode())[numpy.newaxis]
                index = numpy.argmax(self.model.predict(state)[0])
                action = POSSIBLE_ACTIONS[index]
                self.game.act(action)


# Run `python3 player.py` to learn agent and see how it plays
if __name__ == '__main__':
    player = Player()
    player.train(interactive=False)
    player.play(interactive=True)