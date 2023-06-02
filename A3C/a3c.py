import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import worlds_hardest_game as environment
import sys
import pygame
import json

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow.python.keras import layers



parser = argparse.ArgumentParser(description='Run A3C algorithm on the worlds hardest game')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=10, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=10000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()

#vereint Actor Netzwerk und Critic Netzwerk mit durch getrennte Schichten
class ActorCriticModel(keras.Model):
    def __init__(self, action_size):
        super(ActorCriticModel, self).__init__()
        self.action_size = action_size 
        self.flatten = layers.Flatten()
        self.v1 = layers.Dense(256, activation='relu')
        self.v2 = layers.Dense(512, activation='relu')
        self.l1 = layers.Dense(256, activation='relu')
        self.l2 = layers.Dense(512, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        state = self.flatten(inputs)
        v1 = self.v1(state)
        v2 = self.v2(v1)
        l1 = self.l1(state)
        l2 = self.l2(l1)
        logits = self.policy_logits(l2)
        values = self.values(v2)
        return logits, values

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.
    Args:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {episode_reward} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward
    
class MasterAgent():
    def __init__(self):
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        img_height = 20
        img_width = 45
        channels = 3
        self.action_size = 4
        self.opt = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
        
        self.global_model = ActorCriticModel(self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1,img_height,img_width,channels)), dtype=tf.float32))
        
    #erstellt Worker in separaten Threads zum trainieren
    def train(self):
        
        res_queue = Queue()
        workers = [Worker(self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]
        
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        
        moving_average_rewards = []  # record episode reward to plot
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        
        [w.join() for w in workers]
        best = [1]
        for w in workers:
            best_mean = sum(best)/len(best)
            mean = sum(w.reward_history)/len(w.reward_history)
            if mean > best_mean:
                best = w.reward_history
            w.env.close()
        data = np.asarray(Worker.global_score_list)
        data = np.convolve(data, np.ones(50)/50, mode = 'full')
        moving_average = []
        for i in range(len(Worker.global_score_list)):
            moving_average.append(data[i])
        
        
        with open('data/mvg_avg/a3c05_mvg_avg.json', 'w', encoding='utf-8') as f:
            json.dump(moving_average, f, ensure_ascii=False, indent=4)    
            
        with open('data/a3c_05_reward_history.json', 'w', encoding='utf-8') as f:
            json.dump(Worker.global_score_list, f, ensure_ascii=False, indent=4)
            
        with open('data/a3c_05_best_worker.json', 'w', encoding='utf-8') as f:
            json.dump(best, f, ensure_ascii=False, indent=4)    
            
    #Methode, um einen trainierten Agenten das Spiel spielen zu lassen    
    def play(self):
        env = environment.Environment(0)
        state = env.reset()
        model = self.global_model
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format('whg'))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0
        high_score = 0
        
        try:
            while not done:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                        
                policy, value = model(state[None,:])
                policy = tf.nn.softmax(policy)
                action = int(tf.random.categorical(policy, 1))
                state, reward, done = env.step(action)
                if reward > high_score:
                    high_score = reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward, action))
                step_counter += 1
        except KeyboardInterrupt:
            print('Received Keyboard Interrupt. Shutting down.')
        finally:
            env.close()
#Kurzzeitgedächtnis speichert die letzten 10 Erfahrungen            
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    global_score_list = []
    best_score = 0
    save_lock = threading.Lock()
    
    def __init__(self, action_size, global_model, opt, result_queue, idx, save_dir='/tmp'):
        super(Worker, self).__init__()
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.action_size)
        self.worker_idx = idx
        self.env = environment.Environment(self.worker_idx)
        self.save_dir = save_dir
        self.reward_history = []
        self.ep_loss = 0.0
        
    def run(self):
        total_step = 1
        mem = Memory()
        #Trainings loop wie in Pseudocode beschrieben
        while Worker.global_episode < args.max_eps:
            
            current_state = self.env.reset()
            mem.clear()
            high_score = 0.
            max_ep_steps = 1000
            self.ep_loss = 0

            time_count = 0
            done = False
            for ep_steps in range(max_ep_steps):
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                        
                logits, _ = self.local_model(current_state[None,:])
                
                probs = tf.nn.softmax(logits)
                action = int(tfp.distributions.Categorical(probs=probs).sample())
                new_state, reward, done = self.env.step(action)
                if reward > high_score:
                    high_score = reward
                    
                mem.store(current_state, action, reward)
                
                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       args.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())
                    
                    mem.clear()
                    time_count = 0
                    
                    if done: 
                        break
                        
                current_state = new_state
                time_count += 1
                total_step += 1
                    
            # done and print information        
            Worker.global_moving_average_reward = record(Worker.global_episode, high_score, self.worker_idx,
                        Worker.global_moving_average_reward, self.result_queue,
                        self.ep_loss, ep_steps)
            # We must use a lock to save our model and to print to prevent data races.
            if high_score > Worker.best_score:
                with Worker.save_lock:
                    print("Saving best model to {}, "
                          "episode score: {}".format(self.save_dir, high_score))
                    self.global_model.save_weights(
                        os.path.join(self.save_dir,
                                     'model_{}.h5'.format('whg'))
                    )
                    Worker.best_score = high_score
            Worker.global_episode += 1
            Worker.global_score_list.append(high_score)
            self.reward_history.append(high_score)
                    
                
        self.result_queue.put(None)
    
    def compute_loss(self,
                    done,
                    new_state,
                    memory,
                    gamma = 0.99):
        if done:
            reward_sum = 0. #terminal
        else:
            reward_sum = self.local_model(
                  tf.convert_to_tensor(new_state[None, :],
                                       dtype=tf.float32))[-1].numpy()[0]
        
        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]: #reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        
        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                dtype=tf.float32))
        #Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards),
                                        dtype=tf.float32) - values
        #Value loss
        value_loss = advantage ** 2
        
        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        log_policy = tf.math.log(policy+0.000001)
        entropy = tf.reduce_sum(tf.math.multiply(policy, -log_policy))
        
        action_probs = tfp.distributions.Categorical(policy)
        policy_loss = -action_probs.log_prob(memory.actions)
        policy_loss = tf.expand_dims(policy_loss, -1)
       

        #policy_loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,logits=log_policy)
        policy_loss *= advantage
        policy_loss -= 0.05 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss
    
if __name__ == '__main__':
    print(args)
    master = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()

