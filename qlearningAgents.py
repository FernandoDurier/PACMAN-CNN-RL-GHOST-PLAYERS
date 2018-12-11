# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from pacman import Directions
from learningAgents import ReinforcementAgent
from featureExtractors import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.optimizers import RMSprop, Adagrad

import numpy as np

import random,util,math
from collections import deque


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"

        self.Q = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0

        return max([
            self.getQValue(state, action)
            for action in legalActions
        ])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        
        q_max = max([self.getQValue(state, action) for action in legalActions])
        try:
            return random.choice([
                action
                for action in legalActions
                if self.getQValue(state, action) == q_max
            ])
        except:
            l = [action for action in legalActions if self.getQValue(state, action) == q_max]


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        "*** YOUR CODE HERE ***"

        if not legalActions:
            return None

        #Explore
        if random.random() < self.epsilon:
            return random.choice(legalActions)
        #Follow Policy    
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        q_max = self.computeValueFromQValues(nextState)

        self.Q[(state, action)] = self.getQValue(state, action) + (
            self.alpha * (
                reward 
                + self.discount * q_max 
                - self.getQValue(state, action)
            )
        )


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        w = self.getWeights()

        return sum([w[key] * f for key, f in features.items()])


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        q_max = self.computeValueFromQValues(nextState)
        diff = (
            (reward + self.discount * q_max )
            - self.getQValue(state, action)
        )
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        for key, f in features.items():
            weights[key] = weights[key] + self.alpha * diff * f


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class NeuralNetQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', *args, **kwargs):
        PacmanQAgent.__init__(self, *args, **kwargs)

        model = Sequential()
        model.add(Dense(256, init='lecun_uniform', input_shape=(1320,)))
        model.add(Activation('relu'))

        model.add(Dense(64, init='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(5, init='lecun_uniform'))
        model.add(Activation('softmax'))

        opt = Adagrad(lr=self.alpha)
        model.compile(loss='mse', optimizer=opt)

        self.model = model
        self.memory = deque(maxlen=2000)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        qs = [self.getQValue(state, action) for action in legalActions]
        q_max = max(qs)
        return random.choice([
            action
            for action in legalActions
            if self.getQValue(state, action) == q_max
        ])

    def getQValue(self, state, action):
        x = np.zeros(shape=(1, 1320))
        x[0] = self.transformState(state)
        value = self.model.predict(
            x,
            batch_size=1,
        )
        return value[0][self.transformAction(action)]


    def update(self, state, action, nextState, reward):
        self.memory.append((state, action, nextState, reward))
        batch_size = min(len(self.memory), 32)

        for state, action, nextState, reward in random.sample(self.memory, batch_size):
            q_max = self.computeValueFromQValues(nextState)
            y_true = reward + (self.discount * q_max)
            x = np.zeros(shape=(1, 1320))
            x[0] = self.transformState(state)
            y_pred = self.model.predict(x)
            y_pred[0][self.transformAction(action)] = y_true

            self.model.fit(x, y_pred, epochs=1, verbose=0)

    def transformAction(self, action):
        if action == Directions.WEST:
            return 0
        if action == Directions.EAST:
            return 1
        if action == Directions.NORTH:
            return 2
        if action == Directions.SOUTH:
            return 3
        if action == Directions.STOP:
            return 4

    def transformState(self, state):
        shape = (
            state.data.layout.height,
            state.data.layout.width,
        )

        walls = np.array(
            map(
                lambda row: map(int, row),
                state.getWalls().data
            ),
            dtype=np.int8,
        ).T #transpose

        food = np.array(
            map(
                lambda row: map(int, row),
                state.getFood().data
            ),
            dtype=np.int8,
        ).T #transpose

        pacman = np.zeros(shape, dtype=np.int8)
        pos_x, pos_y = state.getPacmanPosition()
        pacman[pos_y][pos_x] = 1

        ghosts = np.zeros(shape, dtype=np.int8)
        scared_ghosts = np.zeros(shape, dtype=np.int8)
        for ghost in state.getGhostStates():
            pos_x, pos_y = ghost.getPosition()
            pos_x = int(pos_x)
            pos_y = int(pos_y)
            if ghost.scaredTimer > 0:
                scared_ghosts[pos_y][pos_x] = 1
            else:
                ghosts[pos_y][pos_x] = 1

        capsules = np.zeros(shape, dtype=np.int8)
        for capsule in state.getCapsules():
            pos_x, pos_y = capsule
            capsules[pos_y][pos_x] = 1

        transformed_state = np.concatenate((walls, pacman, food, capsules, ghosts, scared_ghosts), axis=None)

        return transformed_state
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
