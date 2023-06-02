import numpy as np
import random
import time

def getState(row, col):
    if row == 0:
        return col-9
    
    elif 0<row<6:
        return col+1 + 8*(row-1)
        
    else:
        return col+43

def takeAction(action, row, col, mapstate):
    done = False
    reward = 0
    if actions[action] == "up":
        if row > 0:
            if mapstate[row-1][col]!= "O":
                row-=1
        
    elif actions[action] == "down":
        if row < len(mapstate)-1:
            if mapstate[row+1][col] != "O":
                row+=1
        
    elif actions[action] == "left":
        if col > 0:
            if mapstate[row][col-1] != "O":
                col -=1
        
    elif actions[action] == "right":
        if col < len(mapstate[row])-1:
            if mapstate[row][col+1] != "O":
                col+=1
                
    if mapstate[row][col] == "B":
        done = True
        reward = -1
    
    elif mapstate[row][col] == "G":
        done = True
        reward = 1
        
    else:
        done = False 
        reward = 0
        
    new_state = getState(row,col)
    
    return new_state, reward, done, row, col
    

#Die mapstates stehen für jeweils eine Abbildung für jede Iteration des Levels als Toy-Text
mapstates = {
    0: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","B", "K","K", "K","K", "K","K", "K", "O", "O"],
        2: ["O", "O","K", "K","K", "K","K", "K","K", "B", "O", "O"],
        3: ["O", "O","B", "K","K", "K","K", "K","K", "K", "O", "O"],
        4: ["O", "O","K", "K","K", "K","K", "K","K", "B", "O", "O"],
        5: ["O", "O","B", "K","K", "K","K", "K","K", "K", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    },
    
    1: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","K", "B","K", "K","K", "K","K", "K", "O", "O"],
        2: ["O", "O","K", "K","K", "K","K", "K","B", "K", "O", "O"],
        3: ["O", "O","K", "B","K", "K","K", "K","K", "K", "O", "O"],
        4: ["O", "O","K", "K","K", "K","K", "K","B", "K", "O", "O"],
        5: ["O", "O","K", "B","K", "K","K", "K","K", "K", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    },
    
    2: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","K", "K","B", "K","K", "K","K", "K", "O", "O"],
        2: ["O", "O","K", "K","K", "K","K", "B","K", "K", "O", "O"],
        3: ["O", "O","K", "K","B", "K","K", "K","K", "K", "O", "O"],
        4: ["O", "O","K", "K","K", "K","K", "B","K", "K", "O", "O"],
        5: ["O", "O","K", "K","B", "K","K", "K","K", "K", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    },
    
    3: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","K", "K","K", "B","K", "K","K", "K", "O", "O"],
        2: ["O", "O","K", "K","K", "K","B", "K","K", "K", "O", "O"],
        3: ["O", "O","K", "K","K", "B","K", "K","K", "K", "O", "O"],
        4: ["O", "O","K", "K","K", "K","B", "K","K", "K", "O", "O"],
        5: ["O", "O","K", "K","K", "B","K", "K","K", "K", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    },
    4: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","K", "K","K", "K","B", "K","K", "K", "O", "O"],
        2: ["O", "O","K", "K","K", "B","K", "K","K", "K", "O", "O"],
        3: ["O", "O","K", "K","K", "K","B", "K","K", "K", "O", "O"],
        4: ["O", "O","K", "K","K", "B","K", "K","K", "K", "O", "O"],
        5: ["O", "O","K", "K","K", "K","B", "K","K", "K", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    },
    
    5: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","K", "K","K", "K","K", "B","K", "K", "O", "O"],
        2: ["O", "O","K", "K","B", "K","K", "K","K", "K", "O", "O"],
        3: ["O", "O","K", "K","K", "K","K", "B","K", "K", "O", "O"],
        4: ["O", "O","K", "K","B", "K","K", "K","K", "K", "O", "O"],
        5: ["O", "O","K", "K","K", "K","K", "B","K", "K", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    },
    
    6: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","K", "K","K", "K","K", "K","B", "K", "O", "O"],
        2: ["O", "O","K", "B","K", "K","K", "K","K", "K", "O", "O"],
        3: ["O", "O","K", "K","K", "K","K", "K","B", "K", "O", "O"],
        4: ["O", "O","K", "B","K", "K","K", "K","K", "K", "O", "O"],
        5: ["O", "O","K", "K","K", "K","K", "K","B", "K", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    },
    
    7: {
        0: ["O", "O","O", "O","O", "O","O", "O","O", "K", "K", "G"],
        1: ["O", "O","K", "K","K", "K","K", "K","K", "B", "O", "O"],
        2: ["O", "O","B", "K","K", "K","K", "K","K", "K", "O", "O"],
        3: ["O", "O","K", "K","K", "K","K", "K","K", "B", "O", "O"],
        4: ["O", "O","B", "K","K", "K","K", "K","K", "K", "O", "O"],
        5: ["O", "O","K", "K","K", "K","K", "K","K", "B", "O", "O"],
        6: ["S", "K","K", "O","O", "O","O", "O","O", "O", "O", "O"]
    }
}


actions = ["up", "down", "left", "right"]
state_space_size = 46

#Definition der Hyperparameter und Initialisierung der Q-Tabelle
action_space_size = len(actions)
q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 100000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1 
max_exploration_rate = 1
min_exploration_rate = 0.0001
exploration_decay_rate = 0.0001

rewards_all_episodes = []
if __name__ == '__main__':
    # Q-learning Loop, wie in Pseudocode beschrieben
    for episode in range(num_episodes):

        done = False
        row = 6
        col = 0
        state = getState(row, col)
        rewards_current_episode = 0
        current_mapstate = 0
        mapstate_change = 1

        for step in range(max_steps_per_episode):

            #wechsele den mapstate zu jeder Iteration
            mapstate = mapstates[current_mapstate]

            if step % (len(mapstates)-1) == 0 and current_mapstate > 0:
                mapstate_change = -1
            elif step % (len(mapstates)-1) == 0 and current_mapstate == 0:
                mapstate_change = 1

            current_mapstate += mapstate_change

            #Epsilon greedy Strategie für Aktionsauswahl
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = random.randint(0,3)

            new_state, reward, done, row, col = takeAction(action, row, col, mapstate)
            #Update Q-Tabelle
            q_table[state, action] = q_table[state, action]* (1-learning_rate) +                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
            state = new_state
            rewards_current_episode += reward

            if done == True:
                break

        # verringere Epsilon
        exploration_rate = min_exploration_rate +         (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate* episode)

        
        rewards_all_episodes.append(rewards_current_episode)

    # Print Information
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000
    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000