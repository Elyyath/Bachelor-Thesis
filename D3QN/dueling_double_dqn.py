#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygame
from freegames import floor, vector
import sys
import os
import math
import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import json

#Deep Q-Netzwerk mit 2 versteckten Schichten und 2 verschiedenen Outputs V, A
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, img_height, img_width, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=256)
        self.fc2 = nn.Linear(256, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.flatten(start_dim=1)
        flat1 = F.relu(self.fc1(state))
        flat1 = F.relu(self.fc2(flat1))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))    

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state', 'done')
)

#speichert Erfahrungen bis zu einer maximalen Kapazität und stellt zufällige Batches bereit
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
         
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start- self.end) *             math.exp(-1. * current_step * self.decay)
    
# Diese Klasse wurde in veränderter Form übernommen von
#https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/dueling_ddqn_torch.py
class Agent():
    def __init__(self, strategy, num_actions, gamma, lr, img_height, img_width, batch_size,
                 target_update=5000, chkpt_dir='tmp'):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.chkpt_dir = chkpt_dir
        self.target_update_cnt = target_update
        self.rate = 1
        
        self.memory = ReplayMemory(200000)
        
        self.policy_net = DuelingDeepQNetwork(self.lr, self.num_actions,
                                              img_height=self.img_height, img_width=self.img_width, 
                                            name='dueling_ddqn_policy_net', chkpt_dir=self.chkpt_dir)
        
        self.target_net = DuelingDeepQNetwork(self.lr, self.num_actions,
                                              img_height=self.img_height, img_width=self.img_width, 
                                            name='dueling_ddqn_target_net', chkpt_dir=self.chkpt_dir)
        
    #Aktionsauswahl mit epsilon-greedy Strategie
    def select_action(self, state):
        self.rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        
        if self.rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.policy_net.device) # explore
        else:
            with torch.no_grad():
                _, action_values = self.policy_net.forward(state)
                return action_values.argmax(dim=1).to(self.policy_net.device) # exploit
            
    def update_target_network(self):
        if self.current_step % self.target_update_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    #extrahiert die Bestandteile der Erfahrungen eines Batches als einzelne Batches        
    def extract_tensors(self, experiences):
        batch = Experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)
        t5 = torch.cat(batch.done)

        return (t1, t2, t3, t4, t5)
    
    def save_models(self):
        self.policy_net.save_checkpoint()
        self.target_net.save_checkpoint()
        
    def load_models(self):
        self.policy_net.load_checkpoint()
        self.target_net.load_checkpoint()
    #Methode, zur Durchführung des D3QN Lernalgorithmus wie in Pseudocode beschrieben   
    def learn(self):
        if self.memory.can_provide_sample(self.batch_size):
            
            self.policy_net.optimizer.zero_grad()
            
            self.update_target_network()
            
            experiences = self.memory.sample(self.batch_size)
            
            states, actions, rewards, next_states, dones = self.extract_tensors(experiences)
            
            indices = np.arange(self.batch_size)
            
            state_values, action_values = self.policy_net.forward(states)
            next_state_values, next_action_values = self.target_net.forward(next_states)
            
            eval_next_state_values, eval_next_action_values = self.policy_net.forward(next_states)
            #Berechnung der Q-Werte wie in Gleichung (5) beschrieben
            q_pred = torch.add(state_values, 
                                  (action_values - action_values.mean(dim=1, keepdim=True)))[indices, actions]
            
            q_next = torch.add(next_state_values,
                                  (next_action_values - next_action_values.mean(dim=1, keepdim=True)))
            
            q_eval = torch.add(eval_next_state_values,
                                  (eval_next_action_values - eval_next_action_values.mean(dim=1, keepdim=True)))
            
            max_actions = torch.argmax(q_eval, dim=1)
            
            dones = dones > 0
            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next[indices, max_actions]
            
            loss = self.policy_net.loss(q_target, q_pred).to(self.policy_net.device)
            loss.backward()
            self.policy_net.optimizer.step()
            
#Klasse zur einfacheren Kommunikation zwischen Agent und Environment            
class EnvManager():
    def __init__(self, device):
        self.device = device
        self.env = Environment()
        self.env.reset()
        self.current_screen = None
        self.done = False
        
    def reset(self):
        self.env.reset()
        self.current_screen = None
        
    def close(self):
        self.env.close()
    
    def render(self):
        return self.env.render()
    
    def num_actions_available(self):
        return len(self.env.action_space)
    
    def take_action(self, action):
        reward, self.done = self.env.step(action.item())
                
        return torch.tensor([reward], device = self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    def get_processed_screen(self):
        
        screen = self.render().transpose((2,1,0)) # PyTorch erwartet (channel, height, width)
        return self.transform_screen_data(screen)

    #bringt die Bilddaten in die Form, die das Netzwerk erwartet
    def transform_screen_data(self, screen):       
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((20,45))
            ,T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device) # fügt eine batch Dimension hinzu (batch, channel, height, width)
    
class Environment():
    def __init__(self):
        self.tiles = {
            0: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            1: [0,2,2,2,0,0,0,0,0,0,0,0,1,1,4,4,4,0],
            2: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            3: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            4: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            5: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            6: [0,2,2,2,0,1,1,1,1,1,1,1,1,0,4,4,4,0],
            7: [0,2,2,2,1,1,0,0,0,0,0,0,0,0,4,4,4,0],
            8: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        }
        self.player = vector(50,80)
        self.action_space = ['up', 'down', 'left', 'right']
        self.balls = [
            [vector(110, 50),vector(4,0)],
            [vector(250, 70),vector(-4,0)],
            [vector(110, 90),vector(4,0)],
            [vector(250, 110),vector(-4,0)],
            [vector(110, 130),vector(4,0)]
        ]

        self.done = False
        self.delay = False
        self.screen  = pygame.display.set_mode([350,175])
        self.createMap()
    
    def close(self):
        pygame.quit()
        
    #zeichnet das Level auf Basis der Tilemap
    def createMap(self):
        for row in range(len(self.tiles)):
            for col in range(len(self.tiles[row])):
                if self.tiles[row][col] >0:
                    if self.tiles[row][col] % 2 == 0:
                        color= (76, 230, 41)
                    else:
                        if (row % 2 == 0 and col % 2 == 1) or (row % 2 == 1 and col % 2 == 0):
                            color=(156, 156, 156)
                        else:
                            color = (255, 255, 255)

                    xpos = col * 20
                    ypos = row * 20
                    pygame.draw.rect(self.screen, color ,(xpos,ypos, 20,20))


        self.createObjects()
        pygame.display.update()
    
    #zeichnet die Spielfigur und die Bälle    
    def createObjects(self):

        for ball, course in self.balls:
            pygame.draw.circle(self.screen, (0,0,255),ball, 5)

        pygame.draw.rect(self.screen, (255,0,0), (self.player.x,self.player.y, 10,10))
        
    
    def reset(self):
        self.player = vector(50,80)
        self.balls = [
            [vector(110, 50),vector(4,0)],
            [vector(250, 70),vector(-4,0)],
            [vector(110, 90),vector(4,0)],
            [vector(250, 110),vector(-4,0)],
            [vector(110, 130),vector(4,0)]
        ]
        self.done = False
        self.createMap()
        
    def render(self):
        data = pygame.surfarray.array3d(self.screen)
        return data
    
    #ermittelt, auf welchem Tile die Spielobjekte sich befinden
    def offset(self, position):
        row = int(floor(position.y, 20)/20)
        col = int((floor(position.x, 20)/20))
        return row, col

    #Funktion, die für das Feedback der Umgebung zuständig ist
    #berechnet die Höhe der Belohnung und prüft, ob die gewählte Aktion ausgeführt werden kann
    #prüft auch auf Game-Over
    def validate(self, position):
        
        valid_move = True
        done = False
        reward = 0
        
        #obere, linke Ecke
        row,col = self.offset(position)

        if self.tiles[row][col] == 0:
            valid_move = False
        
            
            
        #obere, rechte Ecke
        row, col = self.offset(vector(position.x + 9, position.y))

        if self.tiles[row][col] == 0:
            valid_move = False
                
        
        #untere, linke Ecke
        row, col = self.offset(vector(position.x, position.y + 9))

        if self.tiles[row][col] == 0:
            valid_move = False
            
        

        #untere, rechte Ecke       
        row, col = self.offset(position+9)

        if self.tiles[row][col] == 0:
            valid_move = False
            
        if self.tiles[row][col] == 4:
            reward = 500
            done = True
            
        if self.tiles[row][col] == 1:
            
            score =  250 - abs(position-vector(270,25)) 
            if score > reward:
                reward = score
        
        if self.tiles[row][col] == 2:
            score = 0.05 * abs(position-vector(20,20))
            if score > reward:
                reward = score
                

        return valid_move, done, reward
    
    
    #wenn False wechselt der Ball die Bewegungsrichtung  
    def validBallPosition(self, position):
        row,col = self.offset(position)

        if self.tiles[row][col] == 0:
            return False

        row, col = self.offset(position+4)

        if self.tiles[row][col] == 0:
            return False    


        row, col = self.offset(position-4)

        if self.tiles[row][col] == 0:
            return False   

        return True
    
    def collision(self):
    
        playerpos = vector(self.player.x+5,self.player.y+5)
        for ball, course in self.balls:

            if abs(ball - playerpos) < 10:
                return True

        return False
    
    
    #führt die gewählte Aktion in der Umgebung aus
    def step(self, action):
        
        if self.action_space[action] == 'up':
            direction = vector(0, 3)
            
        elif self.action_space[action] == 'down':
            direction = vector(0, -3)
            
        elif self.action_space[action] == 'left':
            direction = vector(-3, 0)
        
        else:
            direction = vector(3, 0)
        
        
        valid_move, done, reward = self.validate(self.player + direction)
        
        if valid_move:
             self.player.move(direction)

        
        for ball, course in self.balls:

            position = vector(ball.x+course.x,ball.y+course.y)
            if self.validBallPosition(position):
                ball.move(course)
            else:
                course.x = course.x*(-1)
                ball.move(course)
        
        if self.collision():
            done = True
            reward = -50
            
        self.screen.fill((0,0,0))    
        self.createMap()
        if self.delay == True:
            time.sleep(0.015)
        
        return reward, done

if __name__ == '__main__':
    #Definition der Hyperparameter und Initialisierung aller Objekte
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.000005
    gamma = 0.9
    lr = 0.001
    batch_size = 256
    load_checkpoint = False
    reward_history = []
    epsilon_history = []
    total_steps = 0
    num_episodes = 10000
    max_steps_per_episode = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    em = EnvManager(device)

    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(strategy=strategy, num_actions=em.num_actions_available(), gamma=gamma, 
                    lr=lr, img_height=em.get_screen_height(), img_width=em.get_screen_width(),
                     batch_size=batch_size)

    if load_checkpoint:
        agent.load_models()
        
    #main-loop, wie in Pseudocode beschrieben
    for episode in range(num_episodes):

        em.reset()
        state = em.get_state()
        highest_reward = 0

        for timestep in range(max_steps_per_episode):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            action = agent.select_action(state)
            reward = em.take_action(action)

            if reward > highest_reward:
                highest_reward = reward

            next_state = em.get_state()
            done = em.done
            terminal =  torch.tensor([int(done)], device=device)

            agent.memory.push(Experience(state, action, reward, next_state, terminal))
            agent.learn()

            state = next_state

            if done:
                break

        reward_history.append(highest_reward)
        epsilon_history.append(agent.rate)

        if episode > 0 and episode % 10 == 0:
            agent.save_models()

    em.close()
    #speichert die Daten in JSON Dateien zur späteren Auswertung
    new_data = []
    for data in reward_history:
        new_data.append(data.item())

    with open('data/d3qn_reward_history_04.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    #berechnung des gleitenden Durchschnitts    
    data = np.asarray(new_data)
    data = np.convolve(data, np.ones(500)/500, mode = 'full')
    moving_avg_data = []
    for i in range(len(new_data)):
        moving_avg_data.append(data[i])
    with open('data/mvg_avg/04_mvg_avg.json', 'w', encoding='utf-8') as f:
        json.dump(moving_avg_data, f, ensure_ascii=False, indent=4)        

