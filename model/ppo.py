import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import numpy as np
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ppo/ppo_clip_agent/PPOClipAgent


# https://www.mathworks.com/help/reinforcement-learning/ug/ppo-agents.html

class PPONetwork():
    def __init__(self,
                nS,
                nA,
                batch_size,
                lr,
                gamma,
                lmbda,
                ppo_epochs):
        
        self.nS = nS
        self.nA = nA
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.ppo_epochs = ppo_epochs

        self.actor = self._create_actor_model()
        self.critic = self._create_critic_model()

        self.memory = []
    
    def _create_actor_model(self):
        model = keras.sequential()
        model.add(keras.layers.dense(24, input_dim=self.ns, activation='relu'))
        model.add(keras.layers.dense(24, activation='relu'))
        model.add(keras.layers.dense(self,nA, activation='softmax'))
        model.compile(
            optimizer=Adam(lr=lr),
            loss=[ppo_loss(
                oldpolicy_probs=oldpolicy_probs,

            )]
        )
        return model

    def _create_critic_model(self):
        model = keras.sequential()
        model.add(keras.layers.Dense(24, input_dim=self.ns, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(1, activation='tanh'))
        model.compile(
            optimizer=Adam(lr=lr),
            loss='mean_squared_error'
        )
        return model

    def get_act_dist(self, state):
       '''Use the actor model to predict an action distribution.''' 
       return self.actor.predict(state)

    def get_action(self, act_dist):
        '''Select randomly an action based on 
        the probability distribution.'''
        return np.random.choice(self.nA, p=act_dist)

    def get_value(self, state):
        return self.critic.predict(state)

    def store(self, state, action, value, mask, reward, act_dist):
        act_onehot = np.zeros(self.nA)
        act_onehot[action] = 1
        self.memory.append(
            (state, action, act_onehot, act_dist, value, mask, reward)
        )

    def _get_delta(self, reward, next_value, mask, q_value):
        '''Reward, mask, and q_value are current iteration's values.'''
        delta = reward + self.gamma * next_value * mask - q_value
        return delta

    def _get_gae(self, delta, mask, gae=0):
        gae = delta + self.gamma * self.lmbda  * mask * gae
        return gae

    def get_advantages(self, last_value):
        returns = []
        gae = 0
        for i in reversed(range(len(self.memory))):
            reward_index = 6
            value_index = 4
            mask_index = 5
            reward = self.memory[i][reward_index] # FIXME: magic number
            next_value = self.memory[i+1][value_index] 
            value = self.memory[i][value_index]
            mask = self.memory[i][mask_index]
            if i == len(self.memory) - 1:
                next_value = last_value

            delta = self._get_delta(reward, next_value, mask, value)
            gae = self._get_gae(delta, mask, gae)
            returns.insert(0, gae + value)

        adv = np.array(returns) - values[:-1] # FIXME: why index -1?
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # FIXME: hyperparamter

    def ppo_loss(self, oldpolicy_probs, advantages):
        

    def learn(self, returns, advantages):
        # train actor and critic models to get their losses
        actor_loss = self.actor.fit(
            epochs=self.ppo_epochs,
            verbose=True)
        critic_loss = self.critic.fit(
            epochs=self.ppo_epochs,
            verbose=True)



