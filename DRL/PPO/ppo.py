import os
import glob
import sys
from tkinter import Image
import numpy as np
import random
from collections import deque
import signal
import time
from datetime import datetime
import pandas as pd
import scipy.signal

# from tensorflow.python.keras.saving.save import load_model
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization, Add, Concatenate, Conv2D
from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.backend as K
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB0 as EfficientNet
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
import tensorflow_probability as tfp
# from keras.utils.visualize_util import plot
from tensorflow.python.keras.utils.vis_utils import plot_model
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')#获取GPU列表
print('----gpus---',gpus)
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)])


# from tensorflow.compat.v1 import ConfigProto            #solution for ERROR :Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR 
# from tensorflow.compat.v1 import InteractiveSession     #
# config = ConfigProto()                                  #
# config.gpu_options.allow_growth = True                  #
# session = InteractiveSession(config=config)             #solution for ERROR :Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR

import matplotlib.pyplot as plt
import wandb
from UE4 import CarEnv

# from tensorflow.python.keras.utils.generic_utils import default

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 隐藏warning

try:
    sys.path.append(glob.glob('D:\pzs\CARLA_0.9.12\WindowsNoEditor\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('../carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('/home/amax/NVME_SSD/Pan.zs/carla0.9.12/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
RANDOMSEED = 1
tf.random.set_seed(RANDOMSEED)
random.seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

RECORD_LOSS = 0
TRAIN = 1
PLOT = 1
GAE = 0

STATE_SIZE = 128
ACTION_DIM = 1
SEGMENTATION = 0
EFFICIENT = 0

BATCH_SIZE = 128
LR_A = 0.0001
LR_C = 0.0002
GAMMA = 0.98
TRAIN_EPISODES = 10000  # total number of episodes for training
MAX_STEPS = 4000  # total number of steps for each episode
ACTOR_UPDATE_STEPS = 5  # actor update steps
CRITIC_UPDATE_STEPS = 5  # critic update steps
EPSILON = 0.2 # ppo-clip parameters
LAMBDA = 0.98

TEST_EPISODES = 10  # total number of episodes for testing
# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

if GAE:
    GAMMA = 0.98

wandb.init(project="DRL_PPO", entity="rickkkkk", reinit=True, name="changeCamera_puish_NoCAR_1017_5.16")
wandb.config.hyper_patamter = {
    "State_size": STATE_SIZE,
    "learning_rate_Actor": LR_A,
    "learning_rate_Critic": LR_C,
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "Actor_uodate_steps":ACTOR_UPDATE_STEPS,
    "Critic_uodate_steps":CRITIC_UPDATE_STEPS,
    "lambda_GAE":LAMBDA,
    "semantic_segmentation" : SEGMENTATION,
    "EfficientNet": EFFICIENT
}


# TENSORBOARD = 0
# LOG_PATH = './log/'
# LOG_PATH = 'D:\\pzs\\code\\ddpg\\DDPG_PER\\log'
# LOG_PATH = '/D:/pzs/code/ddpg/DDPG_PER/log/'

def my_handler(signum, frame):
    global stop
    stop = True
    print("============ S T O P ============")

class Agent(object):
    def __init__(self):
        # sess = tf.compat.v1.Session()
        # tf.compat.v1.keras.backend.set_session(sess)

        # self.sess = sess
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.action_bound = 1
        self.train_count = 0
        self.method = 'clip'
        self.lam = LAMBDA

        if self.method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif self.method == 'clip':
            self.epsilon = EPSILON

        self.actor_state_input, self.actor_model = self.create_actor_model()
        # _, self.actor_old_model = self.create_actor_model()
        
        self.critic_model = self.create_critic_model()

        self.critic_optimizer = tf.keras.optimizers.Adam(LR_C)
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_A)

        # self.actor_opt = tf.optimizers.Adam(LR_A)
        # self.critic_opt = tf.optimizers.Adam(LR_C)

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []

    def create_actor_model(self):
        state_input = Input(shape=(STATE_SIZE,STATE_SIZE,3))
        if EFFICIENT:
            a_5 = EfficientNet(include_top=False, weights=None)(state_input)
        else:
            # a_1 = Conv2D(8, 3, activation='relu', padding='same')(state_input)
            # a_2 = Conv2D(16, 3, activation='relu', padding='same')(a_1)
            # a_3 = Conv2D(32, 3, activation='relu', padding='same')(a_2)
            # a_4 = Conv2D(32, 3, activation='relu', padding='same')(a_3)
            # a_5 = Conv2D(64, 3, activation='relu', padding='same')(a_4)

            a_11 = Conv2D(8, 3, activation='relu', padding='same')(state_input)
            a_1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_11)
            a_22 = Conv2D(16, 3, activation='relu', padding='same')(a_1)
            a_2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_22)
            a_33 = Conv2D(64, 3, activation='relu', padding='same')(a_2)
            a_3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_33)
            a_44 = Conv2D(32, 3, activation='relu', padding='same')(a_3)
            a_4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_44)
            a_55 = Conv2D(8, 3, activation='relu', padding='same')(a_4)
            a_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_55)

        # a_5 = GlobalMaxPooling2D()(a_5)
        state_h0 = Flatten()(a_5)
        h12 = Dense(128,activation='relu')(state_h0)
        h2 = Dense(64,activation='relu')(h12)
        Steering_mean = Dense(1, activation='tanh')(h2)
        Steering_sigma = Dense(1, activation='softplus')(h2)
        # Speed = Dense(1, activation='tanh')(h2)
        # output = Concatenate()([Steering, Speed])
        model = Model(inputs=state_input, outputs=[Steering_mean, Steering_sigma])
        return state_input, model
    
    def create_critic_model(self):
        state_input_c = Input(shape=(STATE_SIZE,STATE_SIZE,3))
        if EFFICIENT:
            c_5 = EfficientNet(include_top=False, weights=None)(state_input_c)
        else:
            # c_1 = Conv2D(8, 3, activation='relu', padding='same')(state_input_c)
            # c_2 = Conv2D(16, 3, activation='relu', padding='same')(c_1)
            # c_3 = Conv2D(32, 3, activation='relu', padding='same')(c_2)
            # c_4 = Conv2D(32, 3, activation='relu', padding='same')(c_3)
            # a_5 = Conv2D(64, 3, activation='relu', padding='same')(c_4)

            a_11 = Conv2D(8, 3, activation='relu', padding='same')(state_input_c)
            a_1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_11)
            a_22 = Conv2D(16, 3, activation='relu', padding='same')(a_1)
            a_2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_22)
            a_33 = Conv2D(64, 3, activation='relu', padding='same')(a_2)
            a_3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_33)
            a_44 = Conv2D(32, 3, activation='relu', padding='same')(a_3)
            a_4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_44)
            a_55 = Conv2D(8, 3, activation='relu', padding='same')(a_4)
            a_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_55)


        # a_5 = GlobalMaxPooling2D()(a_5)
        state_h0 = Flatten()(a_5)
        state_h11 = Dense(128,activation='elu')(state_h0)
        state_h2 = Dense(64,activation='elu')(state_h11)
        output_c = Dense(ACTION_DIM,activation='linear')(state_h2)
        model_c = Model(inputs=state_input_c, outputs=output_c)
        return model_c

    # @tf.function
    def train_actor(self,state, action, adv, old_pi):
        with tf.GradientTape() as tape:
            mean, std = self.actor_model(state)
            pi = tfp.distributions.Normal(mean, std)

            # old_mean, old_std = self.actor_old_model(state)
            # old_pi = tfp.distributions.Normal(old_mean, old_std)

            ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            surr = ratio * adv
            if self.method == 'penalty':  # ppo penalty
                kl = tfp.distributions.kl_divergence(old_pi, pi)
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:  # ppo clip
                loss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv))
        a_gard = tape.gradient(loss, self.actor_model.trainable_weights)
        # self.actor_opt.apply_gradients(zip(a_gard, self.actor_model.trainable_weights))
        self.actor_optimizer.apply_gradients(zip(a_gard, self.actor_model.trainable_weights))

        if self.method == 'kl_pen':
            return kl_mean

        if RECORD_LOSS:
            wandb.log({"Actor_loss": loss})

    def train_critic(self, reward, state):
        
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic_model(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic_model.trainable_weights)
        # self.critic_opt.apply_gradients(zip(grad, self.critic_model.trainable_weights))
        self.critic_optimizer.apply_gradients(zip(grad, self.critic_model.trainable_weights))
        
        if RECORD_LOSS:
            wandb.log({"Critic_loss": loss})

    def update(self):
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        mean, std = self.actor_model(s)
        pi = tfp.distributions.Normal(mean, std)
        adv = r - self.critic_model(s)
        if GAE:
            adv = self.discounted_cumulative_sums(adv, self.gamma * self.lam)
        
        print("==========  I am updating ~~~  ==========")

        # update actor
        if self.method == 'kl_pen':
            for _ in range(ACTOR_UPDATE_STEPS):
                kl = self.train_actor(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            for _ in range(ACTOR_UPDATE_STEPS):
                self.train_actor(s, a, adv, pi)

        # update critic
        for _ in range(CRITIC_UPDATE_STEPS):
            self.train_critic(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()
    
    def get_action(self, state, greedy=False):
        state = state[np.newaxis, :].astype(np.float32)
        mean, std = self.actor_model(state)
        if greedy:
            action = mean[0]
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        return np.clip(action, -self.action_bound, self.action_bound)

    def save_model(self):
        try:
            self.actor_model.save(f"carla_ppo_actor.h5", include_optimizer=False)
            # self.actor_old_model.save(f"carla_ppo_actor_old.h5", include_optimizer=False)
            self.critic_model.save(f"carla_ppo_critic.h5", include_optimizer=False)
            print("------------------------------- save model ----------------------------------")
        except:
            print('---------------------------Can not save model----------------------------------')


    def load_model(self):
        try:
            self.actor_model=load_model(f"carla_ppo_actor.h5")
            # self.actor_old_model=load_model(f"carla_ppo_actor_old.h5")
            self.critic_model=load_model(f"carla_ppo_critic.h5")
            print("------------------------------- load model ----------------------------------")
        except:
            print('---------------------------Can not load model----------------------------------')

    def store_transition(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
    
    def finish_path(self, next_state, done):
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic_model(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()
    
    def discounted_cumulative_sums(self, x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def plot(max_step, rewards):
    if PLOT:
        # clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.title('episode. reward')
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Each Episode Reward')
        plt.savefig('reward_episode.png')
        plt.close('all')
        reward_c = pd.DataFrame(rewards)
        reward_c.to_csv("logs/origin.csv", index=None)
        # plt.show()
    else: return

    # current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # train_log_dir = "logs/"+current_time
    # summary_writer = tf.summary.create_file_writer(train_log_dir)
    # with summary_writer.as_default():
    #     tf.summary.scalar('Main/episode_reward', rewards, step=episodes)
    #     # tf.summary.scalar('Main/episode_steps', steps, step=episode)
    # summary_writer.flush()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, my_handler)    #stop signal
    stop = False
    rewards = []
    final_step = []
    max_step = 2500

    env = CarEnv(SEGMENTATION, STATE_SIZE)
    agent = Agent()

    agent.load_model()
    # adam_a = Adam(learning_rate=LR_A)
    # adam_c = Adam(learning_rate=LR_C)
    # agent.critic_model.compile(loss="mse",optimizer=adam_c, metrics=["acc"])
    # agent.actor_old_model.compile(loss="mse",optimizer=adam_a, metrics=["acc"])
    # agent.actor_model.compile(loss="mse",optimizer=adam_a, metrics=["acc"])

    t0 = time.time()
    if TRAIN :
        for e in range(TRAIN_EPISODES):
            state = env.reset()
            epoch_reward = 0
            step = 0
            # state = state
            # s_t = np.reshape(obs, (STATE_SIZE,STATE_SIZE,3))/255
            for t in range(MAX_STEPS):
                step += 1
                # s_t = np.reshape(s_t, (STATE_SIZE,STATE_SIZE,3))

                # control = env.agent_action()
                # a_predict, action = agent.act(s_t, control)
                action = agent.get_action(state)        
                # action = np.reshape(action, (1,ACTION_DIM))
                state_, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward)
                state = state_ 
                epoch_reward +=reward

                print("episode:{} step:{}  action:{} reward:{}".format(e,step,action,reward))
                if len(agent.state_buffer) >= BATCH_SIZE:
                    agent.finish_path(state_, done)
                    agent.update()
                if done:
                    break

                try:
                    if stop:
                        plot(final_step,rewards)
                        env.des()
                        time.sleep(2)
                        raise ValueError ('stop from keyboard')
                except Exception as e:
                    print(str(e))
                    raise ValueError ('stop from keyboard')

            agent.finish_path(state_, done)
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    e + 1, TRAIN_EPISODES, epoch_reward, time.time() - t0))
    
                # print("episode:{} step:{} predict_action:{} action:{} reward:{}".format(e,step,a_predict,action,reward))
            if e == 0:
                rewards.append(epoch_reward)
            else:
                rewards.append(rewards[-1] * 0.9 + epoch_reward * 0.1)
            final_step.append(step)

            wandb.log({"rewards": rewards[-1], "final_step": step})
            # wandb.log({"final_step": step})

            if e%2000 == 0 and e is not 0:
                plot(final_step,rewards)
            
            env.des()

            # if e%500 == 0 and e is not 0:
            #     agent.save_model()
            if step >= max_step and e > 10:
                # max_step = step
                agent.save_model()
    
    else :
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                state, reward, done, _ = env.step(agent.get_action(state, greedy=True))
                episode_reward += reward
                if done:
                    break

                try:
                    if stop:
                        env.des()
                        time.sleep(2)
                        raise ValueError ('stop from keyboard')
                except Exception as e:
                    print(str(e))
                    raise ValueError ('stop from keyboard')
            env.des()
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))