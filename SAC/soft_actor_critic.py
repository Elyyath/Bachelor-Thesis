#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
from collections import namedtuple
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.normal import Normal
import torchvision.transforms as Transforms
import json
import pygame
from freegames import vector, floor
import sys


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward', 'done')
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



#Critic Netzwerk mit 2 versteckten Schichten   
class CriticNetwork(nn.Module):
    def __init__(self, beta, img_height, img_width, n_actions,name, fc1_dims=256, fc2_dims=512, 
              chkpt_dir='tmp'):
        super(CriticNetwork, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        self.fc1 = nn.Linear(in_features=self.img_height*self.img_width*3, out_features=self.fc1_dims)
        self.fc2 = nn.Linear(in_features=self.fc1_dims, out_features=self.fc2_dims)
        self.q = nn.Linear(in_features=self.fc2_dims, out_features=self.n_actions) 
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    def forward(self, state):
        state = state.flatten(start_dim=1)
        action_value = self.fc1(state)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        
        q = self.q(action_value) #q-Werte für alle möglichen Aktionen
        
        return q
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
#Value Netzwerk mit 2 versteckten Schichten         
class ValueNetwork(nn.Module):
    def __init__(self, beta, img_height, img_width, fc1_dims=256, fc2_dims=512, 
                 name='value', chkpt_dir='tmp'):
        super(ValueNetwork, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        self.fc1 = nn.Linear(in_features=self.img_height*self.img_width*3, out_features=self.fc1_dims)
        self.fc2 = nn.Linear(in_features=self.fc1_dims, out_features=self.fc2_dims)
        self.v = nn.Linear(in_features=self.fc2_dims, out_features=1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        state = state.flatten(start_dim=1)
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        
        v = self.v(state_value) #Bewertung eines gegebenen Zustands (V(s))
        
        return v
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

#Actor Netzwerk mit 2 versteckten Schichten 
#gibt Aktionswahrscheinlichkeiten aus
class ActorNetwork(nn.Module):
    def __init__(self, alpha, img_height, img_width, max_action, name, fc1_dims=256, fc2_dims=512,
                 n_actions=4, chkpt_dir='tmp'):
        super(ActorNetwork, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        
        self.fc1 = nn.Linear(in_features=self.img_height*self.img_width*3, out_features=self.fc1_dims)
        self.fc2 = nn.Linear(in_features=self.fc1_dims, out_features=self.fc2_dims)
        self.action_probs = nn.Linear(in_features=self.fc2_dims, out_features=self.n_actions) 
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        state = state.flatten(start_dim=1)
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        prob = self.action_probs(prob)
        prob = F.softmax(prob, dim=1)
        
        return prob
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
# Diese Klasse wurde in veränderter Form übernommen von  
#https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
class Agent():
    def __init__(self, img_height, img_width, alpha=0.001, beta=0.001, gamma=0.9, n_actions=4, max_size=200000,
                 layer1_size=256, layer2_size=256, tau=0.005, batch_size=256, reward_scale=0.8):
        self.gamma = gamma
        self.tau = tau #Faktor für Soft Update des Target Value Netwerks
        self.memory = ReplayMemory(max_size)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.reparam_noise = 1e-6
        
        self.actor = ActorNetwork(alpha, img_height, img_width, n_actions=n_actions, 
                                  name='actor', max_action=1)
        self.critic_1 = CriticNetwork(beta, img_height, img_width, n_actions=n_actions, 
                                     name='critic_1')
        self.critic_2 = CriticNetwork(beta, img_height, img_width, n_actions=n_actions, 
                                      name='critic_2')
        self.value = ValueNetwork(beta, img_height, img_width, name='value')
        self.target_value = ValueNetwork(beta, img_height, img_width, name='target_value')
        
        self.scale = reward_scale
        self.update_network_parameters(tau=1) #Hard Copy
        
    #zufällige Aktionsauswahl mit Kategorialverteilung    
    def choose_action(self, state):
        action_probs = self.actor.forward(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(Experience(state, action, next_state, reward, done))
        
    def update_network_parameters(self, tau=None):
        
        if tau is None:
            tau = self.tau #soft copy der Netzwerkparameter
        
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() +                     (1-tau)*target_value_state_dict[name].clone()
            
        self.target_value.load_state_dict(value_state_dict)
        
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loafing models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        
    #extrahiert die Bestandteile der Erfahrungen eines Batches als einzelne Batches    
    def extract_tensors(self, experiences):
        batch = Experience(*zip(*experiences))

        t1 = T.cat(batch.state)
        t2 = T.cat(batch.action)
        t3 = T.cat(batch.reward)
        t4 = T.cat(batch.next_state)
        t5 = T.cat(batch.done)

        return (t1,t2,t3,t4, t5)    
    

    #Methode, zur Durchführung des SAC Discrete Lernalgorithmus, wie in Pseudocode beschrieben
    def learn(self):
        
        if not self.memory.can_provide_sample(self.batch_size):
            return
        
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.extract_tensors(experiences)

        value = self.value(states).view(-1)
        next_value = self.target_value(next_states).view(-1)
        dones = dones > 0
        next_value[dones] = 0.0

        action_probs = self.actor.forward(states)
        #noise wird hinzugefügt, um log(0) zu vermeiden
        log_probs = T.log(action_probs + self.reparam_noise)
        q1_new_policy = self.critic_1.forward(states)
        q2_new_policy = self.critic_2.forward(states)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        self.value.optimizer.zero_grad()
        value_target = (action_probs*(critic_value - log_probs)).sum(dim=1)
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actor_target = (action_probs*(log_probs - critic_value)).sum(dim=1)
        actor_loss = T.mean(actor_target)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_target = self.scale*rewards + self.gamma*next_value #Scalingfaktor involviert Entropie in die Loss-Berechnung
        q1_old_policy = self.critic_1(states).gather(1, actions.long().unsqueeze(-1)).view(-1)
        q2_old_policy = self.critic_2(states).gather(1, actions.long().unsqueeze(-1)).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_target)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_target)


        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.actor.optimizer.step()

        self.update_network_parameters()
        
#Klasse zur einfacheren Kommunikation zwischen Agent und Environment            
class EnvManager():
    def __init__(self):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.env = Environment()
        self.env.reset()
        self.current_screen = None
        self.done = False
        
    def reset(self):
        self.env.reset()
        self.current_screen = None
        self.done = False
        
    def close(self):
        self.env.close()
    
    def render(self):
        return self.env.render()
    
    def num_actions_available(self):
        return len(self.env.action_space)
    
    def take_action(self, action):
        reward, self.done = self.env.step(action.item())
                
        return T.tensor([reward], device = self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = T.zeros_like(self.current_screen)
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
        screen = T.from_numpy(screen)

        resize = Transforms.Compose([
            Transforms.ToPILImage()
            ,Transforms.Resize((20, 45))
            ,Transforms.ToTensor()
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
                    pygame.draw.rect(self.screen, color ,(xpos, ypos, 20, 20))


        self.createObjects()
        pygame.display.update()
    
    #zeichnet die Spielfigur und die Bälle    
    def createObjects(self):

        for ball, course in self.balls:
            pygame.draw.circle(self.screen, (0,0,255),ball, 5)

        pygame.draw.rect(self.screen, (255,0,0), (self.player.x, self.player.y, 10, 10))
        
    
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
    #Initialisierung aller Objekte
    em = EnvManager()
    agent = Agent(img_height=em.get_screen_height(), img_width=em.get_screen_width(), n_actions=em.num_actions_available())
    n_episodes = 10000
    max_steps_per_episode = 1000

    best_score = 0
    num_steps = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for episode in range(n_episodes):
        em.reset()
        state = em.get_state()
        high_score = 0
        for timestep in range(max_steps_per_episode):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            action = agent.choose_action(state)

            reward = em.take_action(action)
            next_state = em.get_state()
            done = em.done

            terminal =  T.tensor([int(done)], device=em.device)

            agent.remember(state, action, reward, next_state, terminal)

            if reward.item() > high_score:
                high_score = reward.item()

            if not load_checkpoint:
                agent.learn()
            state = next_state
            if done:
                break

        score_history.append(high_score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        num_steps += timestep+1        

    em.close()
    
    #speichert die Daten in JSON Dateien zur späteren Auswertung
    new_data = []
    for data in score_history:
        new_data.append(data)
    with open('data/sac_04_reward_history.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    #berechnung des gleitenden Durchschnitts    
    data = np.asarray(new_data)
    data = np.convolve(data, np.ones(500)/500, mode = 'full')
    moving_avg_data = []
    for i in range(len(new_data)):
        moving_avg_data.append(data[i])
    with open('data/mvg_avg/01_mvg_avg.json', 'w', encoding='utf-8') as f:
        json.dump(moving_avg_data, f, ensure_ascii=False, indent=4)

