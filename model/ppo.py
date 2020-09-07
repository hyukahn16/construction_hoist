import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras import backend as K
from keras import Model
import numpy as np
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ppo/ppo_clip_agent/PPOClipAgent
# https://www.mathworks.com/help/reinforcement-learning/ug/ppo-agents.html

# FIXME: look at how the self.actor.predict takes its input

class PPONetwork():
    def __init__(self,
                nS,
                nA,
                batch_size,
                lr,
                gamma,
                lmbda,
                ppo_epochs,
                clip,
                critic_discount,
                entropy_beta):
        
        self.nS = nS
        self.nA = nA
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.ppo_epochs = ppo_epochs
        self.clip = clip
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta

        self.actor = self._create_actor_model()
        self.critic = self._create_critic_model()

        self.states = []
        self.actions = []
        self.act_onehots = []
        self.values = []
        self.masks = []
        self.rewards = []
        self.act_dists = []

        self.dummy_n = np.zeros((1, 1, self.nA))
        self.dummy_1 = np.zeros((1, 1, 1))
    
    def _create_actor_model(self):
        state_input = keras.layers.Input(shape=self.nS)
        oldpolicy_probs = keras.layers.Input(shape=(1, self.nA,))
        advantages = keras.layers.Input(shape=(1, 1,))
        rewards = keras.layers.Input(shape=(1, 1,))
        values = keras.layers.Input(shape=(1, 1,))

        x = Dense(24, activation='relu', name='fc1')(state_input)
        x = Dense(24, activation='relu', name='fc2')(x)
        out_actions = Dense(self.nA, activation='softmax', name='predictions')(x)

        model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                    outputs=[out_actions])
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss=[self._ppo_loss(
                oldpolicy_probs=oldpolicy_probs,
                advantages=advantages,
                rewards=rewards,
                values=values)
            ]
        ) 
        return model

    def _create_critic_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.nS, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(1, activation='tanh'))
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss='mean_squared_error'
        )
        return model

    def get_act_dist(self, state):
       '''Use the actor model to predict an action distribution.''' 
       return self.actor.predict([state, dummy_n, dummy_1, dummy_1, dummy_1])

    def get_action(self, act_dist):
        '''Select randomly an action based on 
        the probability distribution.'''
        return np.random.choice(self.nA, p=act_dist)

    def get_value(self, state):
        return self.critic.predict([state])

    def store(self, state, action, value, mask, reward, act_dist):
        act_onehot = np.zeros(self.nA)
        act_onehot[action] = 1
        self.states.append(state)
        self.actions.append(action)
        self.act_onehots.append(act_onehot)
        self.act_dists.append(act_dist)
        self.values.append(value)
        self.masks.append(mask)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.act_onehots = []
        self.act_dists = []
        self.values = []
        self.masks = []
        self.rewards = []

    def _get_delta(self, reward, next_value, mask, q_value):
        '''Reward, mask, and q_value are current iteration's values.'''
        delta = reward + self.gamma * next_value * mask - q_value
        return delta

    def _get_gae(self, delta, mask, gae=0):
        gae = delta + self.gamma * self.lmbda  * mask * gae
        return gae

    def get_advantages(self, last_value):
        self.values.append(last_value)
        returns = []
        gae = 0
        for i in reversed(range(len(self.states))):
            reward = self.rewards[i]
            next_value = self.values[i+1]
            value = self.values[i]
            mask = self.masks[i]

            delta = self._get_delta(reward, next_value, mask, value)
            gae = self._get_gae(delta, mask, gae)
            returns.insert(0, gae + value)

        adv = np.array(returns) - self.values[:-1] # FIXME: why index -1?
        norm_adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # FIXME: hyperparameter
        return returns, norm_adv

    def _ppo_loss(self, oldpolicy_probs, advantages, rewards, values):
        def loss(y_true, y_pred):
            newpolicy_probs = y_pred
            ratio = K.exp(
                K.log(newpolicy_probs + 1e-10) -
                K.log(oldpolicy_probs + 1e-10)
            )
            p1 = ratio * advantages
            p2 = K.clip(
                ratio, min_value=1-self.clip, max_value=1+self.clip
                ) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            critic_loss = K.mean(K.square(rewards - values))
            total_loss = \
                self.critic_discount * critic_loss + actor_loss - self.entropy_beta * \
                K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
            return total_loss
        return loss

    def learn(self, returns, advantages):
        '''Train actor and critic models and get their losses.'''
        rewards = np.reshape(self.rewards, newshape=(-1, 1, 1))
        values = self.values[:-1]
        returns = np.reshape(returns, newshape=(-1, 1))

        actor_loss = self.actor.fit(
            [self.states, self.act_dists, advantages, rewards, values],
            [(np.reshape(self.act_onehots, newshape=(-1, self.nA)))],
            epochs=self.ppo_epochs,
            verbose=True)
        critic_loss = self.critic.fit(
            [self.states],
            [returns],
            epochs=self.ppo_epochs,
            verbose=True)



