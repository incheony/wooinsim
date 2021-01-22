# import pacman game 
from pacman import Directions
#from pacmanUtils import *
from game import Agent
class PacmanUtils(Agent):

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        if direction == Directions.EAST:
            return 1.
        if direction == Directions.SOUTH:
            return 2.
        if direction == Directions.WEST:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        if value == 1.:
            return Directions.EAST
        if value == 2.:
            return Directions.SOUTH
        if value == 3.:
            return Directions.WEST
			
    def observationFunction(self, state):
        # do observation
        self.terminal = False
        self.observation_step(state)

        return state
		
    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((batch_size, 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.width, self.height
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)
        return observation

    def registerInitialState(self, state): # inspects the starting state
        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.episode_reward = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.episode_number += 1

    def getAction(self, state):
        move = self.getMove(state)

        # check for illegal moves
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move
from game import Agent
import game

# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class DQN(nn.Module):
    def __init__(self, num_inputs=6, num_actions=4):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        return self.fc4(x)
    
#import other libraries
import random
import numpy as np
import time
from collections import deque

#import other libraries
import os
import util
import random
import numpy as np
import time
import sys

from time import gmtime, strftime
from collections import deque



# model parameters
DISCOUNT_RATE = 0.95        # discount factor
LEARNING_RATE = 0.0005      # learning rate parameter
REPLAY_MEMORY = 50000       # Replay buffer 의 최대 크기
LEARNING_STARTS = 300 	    # 300 스텝 이후 training 시작
TARGET_UPDATE_ITER = 400   # update target network

EPSILON_START = 0.8

# model parameters
model_trained = False

GAMMA = 0.95  # discount factor
LR = 0.01     # learning rate

batch_size = 32            # memory replay batch size
memory_size = 50000		   # memory replay size
start_training = 300 	   # start training at this episode
TARGET_REPLACE_ITER = 400  # update network step

epsilon_final = 0.1   # epsilon final
epsilon_step = 7500

'''    
class PacmanDQN(PacmanUtils):
    def __init__(self, args):      
        
        print("Started Pacman DQN algorithm")
        #print(args)
        self.double = args['double']
        self.multistep = args['multistep']
        self.n_steps = args['n_steps']
        self.model = args['model']

        self.trained_model = args['trained_model']
        if self.trained_model:
            mode = "Test trained model"
        else:
            mode = "Training model"
        
        print("=" * 100)
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}     Model : {}".format(
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode, args['model']))
        print("=" * 100)
        
        
        # pytorch parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # init model
        if(model_trained == True):
            self.policy_net = torch.load('pacman_policy_net.pt').to(self.device)
            self.target_net = torch.load('pacman_target_net.pt').to(self.device)
        else:
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)
            
        self.policy_net.double()
        self.target_net.double()            
            
        # init optim
        self.optim = torch.optim.RMSprop(self.policy_net.parameters(), lr=LEARNING_RATE, alpha=0.95, eps=0.01)
        
        
        
        # Target 네트워크와 Local 네트워크, epsilon 값을 설정
        if self.trained_model:  # Test
            # self.YOUR_NETWORK = torch.load(self.model)
            self.epsilon = 0
        else:                   # Train
            self.epsilon = EPSILON_START  # epsilon init value


        # statistics
        self.win_counter = 0       # number of victory episodes
        self.steps_taken = 0       # steps taken across episodes
        self.steps_per_epi = 0     # steps taken in one episodes   
        self.episode_number = 0
        self.episode_rewards =[]  
        
        # init counters
        self.counter = 0
        self.win_counter = 0
        self.memory_counter = 0
        self.local_cnt = 0

        # init counters
        self.counter = 0
        self.win_counter = 0
        self.memory_counter = 0
        self.local_cnt = 0

        # memory replay and score databases
        self.replay_mem = deque()
        
        # Q(s, a)
        self.Q_global = []  
        self.width = 7
        self.height = 7
        
        self.epsilon = EPSILON_START  # epsilon init value

    """
    def predict(self, state): 
        # state를 넣어 policy에 따라 action을 반환 (epsilon greedy)
        # Hint: network에 state를 input으로 넣기 전에 preprocessing 해야합니다.
        #print(state)
        act = np.random.randint(0, 4)  # random value between 0 and 3
        self.action = act # save action
        return act
    """
    def predict(self, state): # epsilon greedy
        random_value = np.random.rand() 
         
        if random_value > self.epsilon: # exploit 
            # get current state
            temp_current_state = torch.from_numpy(np.stack(self.current_state))
            temp_current_state = temp_current_state.unsqueeze(0)
            temp_current_state = temp_current_state.to(self.device)
            
			# get Qsa
            self.Q_found = self.policy_net(temp_current_state)        
            self.Q_found =  self.Q_found.detach().cpu()
            self.Q_found = self.Q_found.numpy()[0]
			
			# store max Qsa
            self.Q_global.append(max(self.Q_found))
			
			# get best_action - value between 0 and 3Rs
            best_action = np.argwhere(self.Q_found == np.amax(self.Q_found))          
			
            if len(best_action) > 1:  # two actions give the same max
                random_value = np.random.randint(0, len(best_action)) # random value between 0 and actions-1
                move = self.get_direction(best_action[random_value][0])
            else:
                move = self.get_direction(best_action[0][0])
        else: # explore
            random_value = np.random.randint(0, 4)  # random value between 0 and 3
            move = self.get_direction(random_value)

        # save last_action
        self.last_action = self.get_value(move)
        self.action = self.last_action
        return move
    
    
    
    def update_epsilon(self):
        # Exploration 시 사용할 epsilon 값을 업데이트.
        self.epsilon = max(epsilon_final, 1.00 - float(self.episode_number) / float(epsilon_step))
        pass
    
    def getMove(self, state): # epsilon greedy
        random_value = np.random.rand() 
         
        if random_value > self.epsilon: # exploit 
            # get current state
            temp_current_state = torch.from_numpy(np.stack(self.current_state))
            temp_current_state = temp_current_state.unsqueeze(0)
            temp_current_state = temp_current_state.to(self.device)
            
			# get Qsa
            self.Q_found = self.policy_net(temp_current_state)        
            self.Q_found =  self.Q_found.detach().cpu()
            self.Q_found = self.Q_found.numpy()[0]
			
			# store max Qsa
            self.Q_global.append(max(self.Q_found))
			
			# get best_action - value between 0 and 3
            best_action = np.argwhere(self.Q_found == np.amax(self.Q_found))          
			
            if len(best_action) > 1:  # two actions give the same max
                random_value = np.random.randint(0, len(best_action)) # random value between 0 and actions-1
                move = self.get_direction(best_action[random_value][0])
            else:
                move = self.get_direction(best_action[0][0])
        else: # explore
            random_value = np.random.randint(0, 4)  # random value between 0 and 3
            move = self.get_direction(random_value)

        # save last_action
        self.last_action = self.get_value(move)
        return move
    
                 
    def step(self, next_state, reward, done):
        # next_state = self.state에 self.action 을 적용하여 나온 state
        # reward = self.state에 self.action을 적용하여 얻어낸 점수.
        if self.last_action is None:
        
        #if self.action is None:
            self.state = self.preprocess(next_state)
        else:
            self.next_state = self.preprocess(next_state)

            self.state = self.next_state
            # get state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(self.state)

            # get reward
            self.current_score = self.getScore(self.state)
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # ate a ghost 
            elif reward > 0:
                self.last_reward = 10.    # ate food 
            elif reward < -10:
                self.last_reward = -500.  # was eaten
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # didn't eat

            if(self.terminal and self.won):
                self.last_reward = 100.
                self.win_counter += 1
            self.episode_reward += self.last_reward

            # store transition 
            transition = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(transition)
            if len(self.replay_mem) > memory_size:
                self.replay_mem.popleft()
            
            # train model
            #self.train()

        
        # next
        self.episode_reward += reward
        self.steps_taken += 1
        self.steps_per_epi += 1
        self.update_epsilon()

        if(self.trained_model == False):
            self.train()
            self.update_epsilon()
            if(self.steps_taken % TARGET_UPDATE_ITER == 0):
                
                # UPDATING target network 
                self.target_net.load_state_dict(self.policy_net.state_dict())
                pass
		
    def train(self):
        # replay_memory로부터 mini batch를 받아 policy를 업데이트
        #if (self.steps_taken > LEARNING_STARTS):
        #    pass
        
        if (self.local_cnt > start_training):
            batch = random.sample(self.replay_mem, batch_size)
            batch_s, batch_r, batch_a, batch_n, batch_t = zip(*batch)
            
            # convert from numpy to pytorch 
            batch_s = torch.from_numpy(np.stack(batch_s))
            batch_s = batch_s.to(self.device)
            batch_r = torch.DoubleTensor(batch_r).unsqueeze(1).to(self.device)
            batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
            batch_n = torch.from_numpy(np.stack(batch_n)).to(self.device)
            batch_t = torch.ByteTensor(batch_t).unsqueeze(1).to(self.device)
            
            # get Q(s, a)
            state_action_values = self.policy_net(batch_s).gather(1, batch_a)

            # get V(s')
            next_state_values = self.target_net(batch_n)

            # Compute the expected Q values                        
            next_state_values = next_state_values.detach().max(1)[0]
            next_state_values = next_state_values.unsqueeze(1)
            
            expected_state_action_values = (next_state_values * GAMMA) + batch_r
            
			# calculate loss
            loss_function = torch.nn.SmoothL1Loss()
            self.loss = loss_function(state_action_values, expected_state_action_values)
            
			# optimize model - update weights
            self.optim.zero_grad()
            self.loss.backward()
            self.optim.step()            

    def reset(self):
        # 새로운 episode 시작시 불러 오는 함수.
        self.last_score = 0
        self.current_score = 0
        self.episode_reward = 0

        self.episode_number += 1
        self.steps_per_epi = 0
    
    def final(self, state):
        # epsidoe 종료시 불려오는 함수. 수정할 필요 없음.
        done = True
        reward = self.getScore(state)
        if reward >= 0: # not eaten by ghost when the game ends
            self.win_counter +=1

        self.step(state, reward, done)
        self.episode_rewards.append(self.episode_reward)
        win_rate = float(self.win_counter) / 500.0
        avg_reward = np.mean(np.array(self.episode_rewards))
		# print episode information
        if(self.episode_number%500 == 0):
            print("Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(self.episode_number,
                                                                    self.win_counter, win_rate, avg_reward, self.epsilon))
            self.win_counter = 0
            self.episode_rewards= []
            if(self.trained_model==False and self.episode_number%1000 == 0):
                # Save model here
                # torch.save(self.YOUR_NETWORK, self.model)
                pass

    def preprocess(self, state):
        # pacman.py의 Gamestate 클래스를 참조하여 state로부터 자유롭게 state를 preprocessing 해보세요.
        result = state.getPacmanPosition()
        return result
    
    
    
    
    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((batch_size, 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.width, self.height
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)
        return observation

    def registerInitialState(self, state): # inspects the starting state
        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.episode_reward = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.episode_number += 1

    def getAction(self, state):
        move = self.getMove(state)

        # check for illegal moves
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move
'''
    
    
    
    
class PacmanDQN(PacmanUtils):
    def __init__(self, args):        
		
        print("Started Pacman DQN algorithm")
        if(model_trained == True):
            print("Model has been trained")
        else:
            print("Training model")

        # pytorch parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# init model
        if(model_trained == True):
            self.policy_net = torch.load('pacman_policy_net.pt').to(self.device)
            self.target_net = torch.load('pacman_target_net.pt').to(self.device)
        else:
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)
		
        self.policy_net.double()
        self.target_net.double()        
        
        # init optim
        self.optim = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.0005, alpha=0.95, eps=0.01)
        
        # init counters
        self.counter = 0
        self.win_counter = 0
        self.memory_counter = 0
        self.local_cnt = 0

        if(model_trained == False):
            self.epsilon = 0.5     # epsilon init value
    
        else:
            self.epsilon = 0.0     # epsilon init value

        # init parameters
        self.width = args['width']
        self.height = args['height']
        self.num_training = args['numTraining']
        
        # statistics
        self.episode_number = 0
        self.last_score = 0
        self.last_reward = 0.
        
		# memory replay and score databases
        self.replay_mem = deque()
        
		# Q(s, a)
        self.Q_global = []  
		
		# open file to store information
        self.f= open("data_dqn.txt","a")

    def getMove(self, state): # epsilon greedy
        random_value = np.random.rand() 
         
        if random_value > self.epsilon: # exploit 
            # get current state
            temp_current_state = torch.from_numpy(np.stack(self.current_state))
            temp_current_state = temp_current_state.unsqueeze(0)
            temp_current_state = temp_current_state.to(self.device)
            
			# get Qsa
            self.Q_found = self.policy_net(temp_current_state)        
            self.Q_found =  self.Q_found.detach().cpu()
            self.Q_found = self.Q_found.numpy()[0]
			
			# store max Qsa
            self.Q_global.append(max(self.Q_found))
			
			# get best_action - value between 0 and 3
            best_action = np.argwhere(self.Q_found == np.amax(self.Q_found))          
			
            if len(best_action) > 1:  # two actions give the same max
                random_value = np.random.randint(0, len(best_action)) # random value between 0 and actions-1
                move = self.get_direction(best_action[random_value][0])
            else:
                move = self.get_direction(best_action[0][0])
        else: # explore
            random_value = np.random.randint(0, 4)  # random value between 0 and 3
            move = self.get_direction(random_value)

        # save last_action
        self.last_action = self.get_value(move)
        return move
           
    def observation_step(self, state):
        if self.last_action is not None:
            # get state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # get reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # ate a ghost 
            elif reward > 0:
                self.last_reward = 10.    # ate food 
            elif reward < -10:
                self.last_reward = -500.  # was eaten
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # didn't eat

            if(self.terminal and self.won):
                self.last_reward = 100.
                self.win_counter += 1
            self.episode_reward += self.last_reward

            # store transition 
            transition = (self.last_state, self.last_reward, self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(transition)
            if len(self.replay_mem) > memory_size:
                self.replay_mem.popleft()
            
            # train model
            self.train()

        # next
        self.local_cnt += 1
        self.frame += 1
		
		# update epsilon
        if(model_trained == False):
            self.epsilon = max(epsilon_final, 1.00 - float(self.episode_number) / float(epsilon_step))

    def final(self, state):
        # Next
        self.episode_reward += self.last_reward

        # do observation
        self.terminal = True
        self.observation_step(state)
        
		# print episode information
        print("Episode no = " + str(self.episode_number) + "; won: " + str(self.won) 
		+ "; Q(s,a) = " + str(max(self.Q_global, default=float('nan'))) + "; reward = " +  str(self.episode_reward) + "; and epsilon = " + str(self.epsilon))
		
		# copy episode information to file
        self.counter += 1
        if(self.counter % 10 == 0):
            self.f.write("Episode no = " + str(self.episode_number) + "; won: " + str(self.won) 
		+ "; Q(s,a) = " + str(max(self.Q_global, default=float('nan'))) + "; reward = " +  str(self.episode_reward) + "; and epsilon = " 
		+ str(self.epsilon) + ", win percentage = " + str(self.win_counter / 10.0) + ", " + str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + "\n")
            self.win_counter = 0

        if(self.counter % 500 == 0):
            # save model
            torch.save(self.policy_net, 'pacman_policy_net.pt')
            torch.save(self.target_net, 'pacman_target_net.pt')
        
        if(self.episode_number % TARGET_REPLACE_ITER == 0):
            print("UPDATING target network")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        if (self.local_cnt > start_training):
            batch = random.sample(self.replay_mem, batch_size)
            batch_s, batch_r, batch_a, batch_n, batch_t = zip(*batch)
            
            # convert from numpy to pytorch 
            batch_s = torch.from_numpy(np.stack(batch_s))
            batch_s = batch_s.to(self.device)
            batch_r = torch.DoubleTensor(batch_r).unsqueeze(1).to(self.device)
            batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
            batch_n = torch.from_numpy(np.stack(batch_n)).to(self.device)
            batch_t = torch.ByteTensor(batch_t).unsqueeze(1).to(self.device)
            
            # get Q(s, a)
            state_action_values = self.policy_net(batch_s).gather(1, batch_a)

            # get V(s')
            next_state_values = self.target_net(batch_n)

            # Compute the expected Q values                        
            next_state_values = next_state_values.detach().max(1)[0]
            next_state_values = next_state_values.unsqueeze(1)
            
            expected_state_action_values = (next_state_values * GAMMA) + batch_r
            
			# calculate loss
            loss_function = torch.nn.SmoothL1Loss()
            self.loss = loss_function(state_action_values, expected_state_action_values)
            
			# optimize model - update weights
            self.optim.zero_grad()
            self.loss.backward()
            self.optim.step() 