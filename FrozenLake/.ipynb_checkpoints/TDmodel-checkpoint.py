"""Temporal-Difference Learning models : SARSA and Q-Learning"""
import gym
import numpy as np
import random

FHGmap4=[list(s) for s in """SFFF
FHFH
FFFH
HFFG""".split('\n')]

FHGmap8=[list(s) for s in """SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFFFHFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG""".split('\n')]


class TDmodel():
    def __init__(self, model, env_size, gamma, alpha, verbose=False, epsilon_greedy=False, epsilon=0.5):
        """
        model: 'SARSA','Q'
        env_size: [4,8] (for the FrozenLake gym)
        gamma: discount facotr
        alpha: float parameter
        verbose: bool
        """

        self.model = model
        self.env_size = env_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon

        if env_size == 4:

            try:
                from gym.envs.registration import register
                register(id='FrozenLakeNotSlippery-v0',
                    entry_point='gym.envs.toy_text:FrozenLakeEnv',
                    kwargs={'map_name' : '4x4', 'is_slippery': False},
                    max_episode_steps=100,
                    reward_threshold=0.8196, # optimum = .8196
                )
            except:
                pass # this breaks if I try to reregister an env

            self.env = gym.make('FrozenLakeNotSlippery-v0')
            self.FHGmap = FHGmap4

        elif env_size == 8:

            try:
                from gym.envs.registration import register
                register(
                    id='FrozenLake8x8NotSlippery-v0',
                    entry_point='gym.envs.toy_text:FrozenLakeEnv',
                    kwargs={'map_name' : '8x8', 'is_slippery': False},
                    max_episode_steps=100,
                    reward_threshold=0.8196, # optimum = .8196
                )
            except:
                pass # this breaks if I try to reregister an env

            self.env = gym.make('FrozenLake8x8NotSlippery-v0')
            self.FHGmap = FHGmap8 

        self.Q = np.zeros([self.env.observation_space.n,self.env.action_space.n]) # Q_values
        self.V = np.zeros([self.env.observation_space.n]) # Values for states 
        self.P = np.zeros([self.env.observation_space.n]) # Policy

        #this are a backlog for training, used later 
        self.states = []
        self.rewards = []
        self.l1_norm = []
        self.V_log = []
        self.paths = []

    def choose_action(self, state, epsilon=0.5, epsilon_greedy=False):
        """
        I choose an action based on the current policy, that is based on the current Q values

        this is equivalent to computing Pi(n)(s)

        Args: state
        return: action (based on policy)
        """
        if self.epsilon_greedy:
            if random.random() < self.epsilon:
                self.P[state] = np.argmax(self.Q[state])
            else:
                self.P[state] = np.random.choice(np.delete(np.arange(self.env.action_space.n), np.argmax(self.Q[state])))

        elif not self.epsilon_greedy:
            self.P[state] = np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max())) # randomly choose the best actions (for ties)

        return int(self.P[state])


    def sarsa_learn(self, state, next_state, reward, action, verbose=False):
        """
        I updated the value of the current state

        SARSA

        return: update
        """
        next_action = self.choose_action(next_state)

        previous_Q = self.Q[state,action]
        self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
        self.V[state] = max([self.Q[state, action] for action in range(self.env.action_space.n)]) 

        if verbose and previous_Q != self.Q[state,action]: print("[sarsa_learn] state {} : action {} - learnt {}".format(
            state,
            action,
            self.Q[state,action] - previous_Q))
        return next_state, next_action


    def q_learn(self, state, next_state, reward, action, verbose=False):
        """
        I updated the value of the current state

        Q_Learning

        return: update
        """
        previous_Q = self.Q[state,action]
        self.Q[state, action] += self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state, action])
        self.V[state] = max([self.Q[state, action] for action in range(self.env.action_space.n)]) 

        if verbose and previous_Q != self.Q[state,action]: print("[sarsa_learn] state {} : action {} - learnt {}".format(
            state,
            action,
            self.Q[state,action] - previous_Q))
        return next_state, _


    def train(self, num_episodes=None, max_iter_per_episodes=None,verbose=False):
        """
        returns
        V_log, array of V-matrices for heatmap
        rewards, array of rewards
        states, array of ending states
        paths, array of paths for every episode
        """

        from time import time
        start_time = time()

        if num_episodes is not None and max_iter_per_episode is not None:
            # if training vals not provided by function call just run default
            pass
        elif self.env_size == 4:
            num_episodes=2000
            max_iter_per_episode=100
        elif self.env_size == 8:
            num_episodes=20000
            max_iter_per_episode=800

        for i in range(num_episodes):
            if verbose and i % 50 == 0: print(i)
            episode = 0
            state = self.env.reset()
            path = [state]
            done = False
            previous_V = self.V.copy()

            action = self.choose_action(state)

            while episode < max_iter_per_episode:
                #env.render() prints the frozenlake with an indicator showing where the agent is. You can use it for debugging.

                episode+=1        

                # env.step() gives zyou next state, reward, done(whether the episode is over)
                # s1 - new state, r-reward, d-whether you are done or not

                if verbose: print('State : ',state, ' Action : ', action, ' State 1 : ', next_state, ' Reward : ',reward, 'Done : ', done)
                    
                if self.model == "SARSA":
                    next_state, reward, done, _ = self.env.step(action)
                    next_state, next_action = self.sarsa_learn(state, next_state, reward, action)
                    action = next_action
                    state = next_state
                    
                if self.model == "Q":
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state, _ = self.q_learn(state, next_state, reward, action)
                    state = next_state
                
                path.append(next_state)
                    
                if done:
                    if verbose: print('Episode Over')
                    if reward != 1 and verbose: print('Fell into hole with reward: ', reward)
                    self.rewards.append(reward)
                    self.l1_norm.append(np.linalg.norm(self.V-previous_V))
                    self.V_log.append(self.V.copy())
                    self.states.append(state)
                    self.paths.append(path)
                    break

        print("Training time: "+str(time()-start_time))
        return self.V_log, self.rewards, self.states, self.paths

    def plot_training_results(self):

        import seaborn as sns
        from matplotlib import pyplot as plt
        
        plt.figure(figsize=(15,7))
        plt.subplot('221')
        dimension = int(np.sqrt(self.env.observation_space.n))
        self.P = self.P.reshape(dimension,dimension).astype(int)
        for x in range(dimension):
            for y in range(dimension):
                d = 0.25
                dx = {0:-d,1:0,2:d,3:0}[self.P[y, x]]
                dy = {0:0,1:d,2:0,3:-d}[self.P[y, x]]
                plt.arrow(x+.7-dx,y+.5-dy,dx,dy,head_width=0.1,color="black",alpha=0.9)
                plt.text(x + .1 ,y + .7,self.FHGmap[y][x],color="green",size=21,alpha=0.5)
        sns.heatmap(self.V.reshape(dimension,dimension))
        
        plt.subplot('222')
        plt.plot(np.convolve(self.rewards, np.ones((100,))/100, mode='valid'))
        plt.title("100-periods moving average for rewards")
        
        plt.subplot('223')
        plt.plot(np.convolve(self.l1_norm, np.ones((100,))/100, mode='valid'))
        plt.title("L1 Norm of the self.Values vector")





