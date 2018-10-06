#!/usr/local/bin/python3
from numpy.random import randint

from objects import Home, Street, Factory, Shop
from agent import Agent

class Game:
    def __init__(self):
        self.desk_width = 3
        self.desk_height = 3

        # Defining a game desk
        self.desk = [
            [Home(),   Street(), Factory()],
            [Street(), Street(), Street() ],
            [Street(), Street(), Shop()   ],
        ]

        # Total state size
        # state_size: (i, j) + (food, rest, creds)
        self.state_size = 2 + 3

    def create_agent(self):
        self.turns = 0
        self.agent = Agent(food=10, rest=10, creds=10)
        i, j = randint(self.desk_height), randint(self.desk_width)
        self.agent.set_position(i, j)
        self.desk[i][j].populate()

    def encode(self):
        return self.agent.get_position() + (self.agent.food, self.agent.rest, self.agent.creds)
    
    def decode(self):
        pass

    def show(self):
        for row in self.desk:
            for game_object in row:
                game_object.show()
            print('')
        
        print('F: %i R: %i $: %i TURN: %i' % (self.agent.food, self.agent.rest, self.agent.creds, self.turns))

    def is_over(self):
        return self.agent.is_dead()
    
    def is_action_valid(self, action):
        i, j = self.agent.get_position()
        return (0 <= i + action.di < self.desk_height) and (0 <= j + action.dj < self.desk_width)

    def act(self, action):
        self.turns += 1
        confidence = self.agent.get_confidence()

        i, j = self.agent.get_position()
        if self.is_action_valid(action):
            self.desk[i][j].depopulate()
            self.desk[i + action.di][j + action.dj].populate()   
            i, j = self.agent.act(action)

        self.desk[i][j].affect(self.agent)
        if self.agent.is_dead():
            self.desk[i][j].depopulate()
            return -10
        else:
            confidence_next = self.agent.get_confidence()
            return confidence_next - confidence