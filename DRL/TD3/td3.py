from ast import While
import os
import glob
import sys
from cv2 import merge
import numpy as np
import random
from collections import deque
import signal
import time
import cv2

# from tensorflow.python.keras.saving.save import load_model
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization, Add, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.backend as K
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB0 as EfficientNet
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')#获取GPU列表
print('----gpus---',gpus)
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*16)])


# from tensorflow.compat.v1 import ConfigProto            #solution for ERROR :Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR 
# from tensorflow.compat.v1 import InteractiveSession     #
# config = ConfigProto()                                  #
# config.gpu_options.allow_growth = True                  #
# session = InteractiveSession(config=config)             #solution for ERROR :Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR

import matplotlib.pyplot as plt

import wandb
from BUFFER import Memory
from UE4 import CarEnv
from Noise import ActionNoise, AdaptiveParamNoiseSpec

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 隐藏warning

# from tensorflow.python.keras.utils.generic_utils import default

try:
    sys.path.append(glob.glob('D:\pzs\CARLA_0.9.12\WindowsNoEditor\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
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

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(1)
np.random.seed(1)

RECORD_LOSS = 0
TRAIN = 1
PLOT = 1
PER = 1

STATE_SIZE = 128
ACTION_DIM = 1
SEGMENTATION = 0
EFFICIENT = 0

BUFF_SIZE = 1000000
TRAIN_EPISODES = 10000
TEST_EPISODES = 10

BATCH_SIZE = 128
LR_A = 0.0001
LR_C = 0.0002
GAMMA = 0.95
TAU = 0.005
TRAIN_ACTOR_EACH_C = 3 # each X+1 train
NOISE_EXPLORE_STD = 0.1
NOISE_POLICY_STD = 0.2
REGULARIZER_L2 = 0.001

EPSILON = 0.999
EPSILON = 0.001
EPSILON_DECAY = 0.999

RANDOM_ACTION = 1
PLOT = 1
# TENSORBOARD = 0
# LOG_PATH = './log/'
# LOG_PATH = 'D:\\pzs\\code\\ddpg\\DDPG_PER\\log'
# LOG_PATH = '/D:/pzs/code/ddpg/DDPG_PER/log/'

if TRAIN:
    wandb.init(project="DRL_TD3", entity="rickkkkk", reinit=True, name="changeCamera_puish_NoCAR_1017_5.19")
    wandb.config.hyper_patamter = {
        "State_size": STATE_SIZE,
        "learning_rate_Actor": LR_A,
        "learning_rate_Critic": LR_C,
        "gamma": GAMMA,
        "batch_size": BATCH_SIZE,
        "Tau":TAU,
        "Train_actor_each_critic":TRAIN_ACTOR_EACH_C,
        "Explore_noise_std":NOISE_EXPLORE_STD,
        "semantic_segmentation" : SEGMENTATION,
        "Policy_noise_std" : NOISE_POLICY_STD,
        "Regularizer_L2": REGULARIZER_L2,
        "PER_IS": PER,
        "EfficientNet": EFFICIENT
    }

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
        self.epsilon_decay = EPSILON_DECAY
        self.tau = TAU
        # self.memory = deque(maxlen=400000)
        self.batch_size = BATCH_SIZE
        self.buff = Memory(BUFF_SIZE, PER)
        self.noise = ActionNoise(mu=0, sigma=1)
        self.train_count = 0

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        # self.actor_critic_grad = tf.placeholder(tf.float32,[None,ACTION_DIM])
        # actor_model_weights = self.actor_model.trainable_weights
        # self.actor_grads = tf.gradients(self.actor_model.output,actor_model_weights,-self.actor_critic_grad)
        # grads = zip(self.actor_grads,actor_model_weights)
        # self.optimize = tf.train.AdamOptimizer(LR_A).apply_gradients(grads)
 
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        # self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        self.critic_state_input_2, self.critic_action_input_2, self.critic_model_2 = self.create_critic_model()
        _, _, self.target_critic_model_2 = self.create_critic_model()
        # self.critic_grads_2 = tf.gradients(self.critic_model_2.output, self.critic_action_input_2)
        
        # self.sess.run(tf.initialize_all_variables())
        # self.sess.run(tf.global_variables_initializer())
        # K.clear_session()

        self.critic_optimizer = tf.keras.optimizers.Adam(LR_C)
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_A)

    def create_actor_model(self):
        state_input = Input(shape=(STATE_SIZE,STATE_SIZE,3))
        steer_input = Input(shape=(ACTION_DIM))
        if EFFICIENT:
            a_5 = EfficientNet(include_top=False, weights=None)(state_input)
        else:
            # a_1 = Conv2D(8, 3, activation='relu')(state_input)
            # a_2 = Conv2D(16, 3, activation='relu')(a_1)
            # a_3 = Conv2D(64, 3, activation='relu')(a_2)
            # a_4 = Conv2D(32, 3, activation='relu')(a_3)
            # a_5 = Conv2D(8, 3, activation='relu')(a_4)

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

            # a_11 = Conv2D(8, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(state_input)
            # a_1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_11)
            # a_22 = Conv2D(16, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_1)
            # a_2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_22)
            # a_33 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_2)
            # a_3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_33)
            # a_44 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_3)
            # a_4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_44)
            # a_55 = Conv2D(8, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_4)
            # a_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_55)
        # aa_22 = GlobalMaxPooling2D()(a_5)
        state_h0 = Flatten()(a_5)
        state_output = Dense(128,activation='relu')(state_h0)    #, kernel_regularizer=l2(REGULARIZER_L2)
        steer_output = Dense(32, activation='relu')(steer_input)
        merge1 = Concatenate()([state_output,steer_output])
        h1 = Dense(512,activation='relu')(merge1)
        h2 = Dense(256,activation='relu')(h1)
        Steering = Dense(1, activation='tanh')(h2)

        # state_h0 = Flatten()(a_5)
        # state_output = Dense(128,activation='relu', kernel_regularizer=l1(REGULARIZER_L2))(state_h0)    #, kernel_regularizer=l2(REGULARIZER_L2)
        # steer_output = Dense(32, activation='relu', kernel_regularizer=l1(REGULARIZER_L2))(steer_input)
        # merge1 = Concatenate()([state_output,steer_output])
        # h1 = Dense(512,activation='relu', kernel_regularizer=l1(REGULARIZER_L2))(merge1)
        # h2 = Dense(256,activation='relu', kernel_regularizer=l1(REGULARIZER_L2))(h1)
        # Steering = Dense(1, activation='tanh', kernel_regularizer=l1(REGULARIZER_L2))(h2)
    
        model = Model(inputs=[state_input, steer_input], outputs=Steering)
        return state_input, model

    def create_critic_model(self):
        state_input_c = Input(shape=(STATE_SIZE,STATE_SIZE,3))
        action_input = Input(shape=(ACTION_DIM))
        if EFFICIENT:
            c_5 = EfficientNet(include_top=False, weights=None)(state_input_c)
        else:
            # c_1 = Conv2D(8, 3, activation='relu')(state_input_c)
            # c_2 = Conv2D(16, 3, activation='relu')(c_1)
            # c_3 = Conv2D(64, 3, activation='relu')(c_2)
            # c_4 = Conv2D(32, 3, activation='relu')(c_3)
            # c_5 = Conv2D(8, 3, activation='relu')(c_4)

            # a_11 = Conv2D(8, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(state_input_c)
            # a_1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_11)
            # a_22 = Conv2D(16, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_1)
            # a_2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_22)
            # a_33 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_2)
            # a_3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_33)
            # a_44 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_3)
            # a_4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_44)
            # a_55 = Conv2D(8, 3, activation='relu', padding='same', kernel_regularizer=l1(REGULARIZER_L2))(a_4)
            # a_5 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(a_55)

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

        # ss_11 = GlobalMaxPooling2D()(c_5)
        
        state_h0 = Flatten()(a_5)
        state_output = Dense(128,activation='relu')(state_h0)
        action_output = Dense(32, activation="relu")(action_input)
        merged_1 = Concatenate()([state_output,action_output])
        h1 = Dense(512, activation='relu')(merged_1)
        h2 = Dense(256, activation='relu')(h1)
        output_c = Dense(ACTION_DIM)(h2)

        # state_h0 = Flatten()(a_5)
        # state_output = Dense(128,activation='relu', kernel_regularizer=l1(REGULARIZER_L2))(state_h0)
        # action_output = Dense(32, activation="relu", kernel_regularizer=l1(REGULARIZER_L2))(action_input)
        # merged_1 = Concatenate()([state_output,action_output])
        # h1 = Dense(512, activation='relu', kernel_regularizer=l1(REGULARIZER_L2))(merged_1)
        # h2 = Dense(256, activation='relu', kernel_regularizer=l1(REGULARIZER_L2))(h1)
        # output_c = Dense(ACTION_DIM, kernel_regularizer=l1(REGULARIZER_L2))(h2)
        # output_c = Dense(ACTION_DIM,activation='tanh')(merged_h1)

        model_c = Model(inputs=[state_input_c,action_input], outputs=output_c)
        return state_input_c, action_input, model_c
 
    def remember(self,s_t,action,reward,s_t1,done):
        experiences = (s_t,action,reward,s_t1,done)
        self.buff.store(experiences)
        # self.memory.append(experiences)
 
    def train(self):
        if self.buff.num < self.batch_size: 
            return

        self.train_count +=1
        tree_idx, samples, self.ISWeights = self.buff.sample(self.batch_size)

        self.samples = samples
        # print(np.shape(samples), samples)\
        self.s_ts, self.actions, self.rewards, self.s_ts1, self.dones = self.stack_samples(self.samples)
        # self.s_ts, self.actions, self.rewards, self.s_ts1, self.dones = self.s_ts.astype(np.float32), self.actions.astype(np.float32), self.rewards.astype(np.float32), self.s_ts1.astype(np.float32), self.dones.astype(np.float32)
        target_actions = self.target_actor_model([self.s_ts1, self.actions], training=True)
        target_actions = tf.convert_to_tensor(self.noise.add_noise(target_actions, std=NOISE_POLICY_STD))
        q_value, q_target = self.train_net(self.s_ts, self.actions, self.rewards, self.s_ts1, self.dones, target_actions)
        if self.train_count >= TRAIN_ACTOR_EACH_C:
            self.train_count = 0
            self.train_actor(self.s_ts, self.actions)

        if PER:
            err_batch = self.update_err(np.asarray(q_value), np.asarray(q_target))
            self.buff.batch_update(tree_idx, err_batch)

    @tf.function
    def train_actor(self,s_ts, actions):
        # predicted_actions = self.actor_model.predict(s_ts)
        # grads = self.sess.run(self.critic_grads,feed_dict={        
        #         self.critic_state_input: s_ts,
        #         self.critic_action_input: predicted_actions
        # })[0]
        # self.sess.run(self.optimize,feed_dict={
        #         self.actor_state_input: s_ts,
        #         self.actor_critic_grad: grads
        # })

        with tf.GradientTape() as tape:
            actions_out = self.actor_model([s_ts, actions], training=True)
            critic_value_a = self.critic_model([s_ts, actions_out], training=True)
            # actor_loss = - 0.1 * tf.math.reduce_mean(critic_value_a)
            actor_loss = - tf.math.reduce_mean(critic_value_a)
            # actor_loss += self.action_l2 * tf.math.reduce_mean(tf.square(critic_value_a / self.max_u))

        # tf.print("=====The actor_loss is ", actor_loss, "=====")
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
        if RECORD_LOSS:
            wandb.log({"Actor_loss": actor_loss})

    @tf.function
    def train_net(self,s_ts, actions, rewards, s_ts1, dones, target_actions):
        # target_actions = self.target_actor_model.predict(s_ts1)
        # target_actions = self.noise.add_noise(target_actions, eval_noise_scale=0.5)
        
        # pre_1 = self.target_critic_model.predict([s_ts1,target_actions])
        # pre_2 = self.target_critic_model_2.predict([s_ts1,target_actions])
        # futurn_rewards = np.zeros(np.shape(pre_1))
        # for i in range(len(pre_1)):
        #     for o in range(len(pre_1[0])):
        #         futurn_rewards[i][o] = min(pre_1[i][o], pre_2[i][o])

        # reward_2 = np.reshape(rewards, (BATCH_SIZE,1))
        # # reward_2 = np.concatenate([reward_2, reward_2], axis=1)
        # done = np.reshape(np.asarray(dones), (BATCH_SIZE,1))
        # # done = np.concatenate([done, done], axis=1)

        # reward_2 += self.gamma * futurn_rewards * (1-done)
        # self.history = self.critic_model.fit([s_ts,actions], reward_2, verbose=1, sample_weight=self.ISWeights)#callbacks=[self.tensorboard_callback]
        # # self.history = self.critic_model_2.fit([s_ts,actions], reward_2, verbose=1, sample_weight=self.ISWeights)#callbacks=[self.tensorboard_callback]

        # q_value_1 = self.critic_model.predict([s_ts,actions])
        # q_value_2 = self.critic_model_2.predict([s_ts,actions])
        # q_value = np.zeros(np.shape(q_value_2))
        # for i in range(len(q_value_1)):
        #     for o in range(len(q_value_1[0])):
        #         q_value[i][o] = min(q_value_1[i][o], q_value_2[i][o])
        target_value = self.target_critic_model([s_ts1, target_actions], training=True)
        target_value_2 = self.target_critic_model_2([s_ts1, target_actions], training=True)
        y = rewards + (1-dones) * self.gamma * tf.minimum(target_value, target_value_2)
        with tf.GradientTape() as tape:
            critic_value = self.critic_model([s_ts, actions], training=True)
            critic_loss = self.ISWeights * tf.math.reduce_mean( tf.math.square(y - critic_value))

        # tf.print("+++++ The critic loss is ", critic_loss, "+++++")

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
        
        with tf.GradientTape() as tape:
            critic_value_2 = self.critic_model_2([s_ts, actions], training=True)
            critic_loss_2 = self.ISWeights * tf.math.reduce_mean( tf.math.square(y - critic_value_2))

        critic_grad_2 = tape.gradient(critic_loss_2, self.critic_model_2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad_2, self.critic_model_2.trainable_variables))
        
        # with tf.GradientTape() as tape:
        #     actions_out = self.actor_model(s_ts, training=True)
        #     critic_value_a = self.critic_model([s_ts, actions_out], training=True)
        #     actor_loss = - tf.math.reduce_mean(critic_value_a)

        # tf.print("=====The actor_loss is ", actor_loss, "=====")
        # actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        # self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
        if RECORD_LOSS:
            wandb.log({"Critic1_loss": critic_loss, "Critic2_loss": critic_loss_2})
            wandb.log({"Critic1_value": critic_value, "Critic2_value": critic_value_2})
            wandb.log({"target_value_2": target_value_2, "target_value": target_value})
    
        return critic_value, y

    def update_target(self):
        self.update_actor_target()
        self.update_critic_target()
        # self.update_target_each(self.target_actor_model.variables, self.actor_model.variables, self.tau)
        # self.update_target_each(self.target_critic_model.variables, self.critic_model.variables, self.tau)
        # self.update_target_each(self.target_critic_model_2.variables, self.critic_model_2.variables, self.tau)
    
    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)
    
    @tf.function
    def update_target_each(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
 
    def update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

        critic_model_2_weights = self.critic_model_2.get_weights()
        critic_target_2_weights = self.target_critic_model_2.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_2_weights[i] = critic_model_2_weights[i]*self.tau + critic_target_2_weights[i]*(1-self.tau)
        self.target_critic_model_2.set_weights(critic_target_2_weights)
 
    def act(self,s_t, control=None, get_action=0):
        # print("========= Control is ===", control.steer)
        a_t = np.zeros([1,ACTION_DIM])
        if self.epsilon > 0.001:
            self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            # control = self.ue4.agent_action()
            a_t[0][0] = np.random.uniform(-1,1)
            # a_t[0][1] = np.random.random()
            # a_t[0][0] = control.steer
            # a_t[0][1] = control.throttle
            return '[[random action]]', a_t[0]
        a_predict = self.actor_model([s_t, get_action])
        return a_predict, self.noise.add_noise(a_predict, std=NOISE_EXPLORE_STD)

    def stack_samples(self, samples):       #maybe
        s_ts = np.array([e[0] for e in samples], dtype='float32')
        actions = np.array([e[1] for e in samples], dtype='float32')
        rewards = np.array([e[2] for e in samples], dtype='float32')
        s_ts1 = np.array([e[3] for e in samples], dtype='float32')
        dones = np.array([e[4] for e in samples], dtype='float32')

        # s_ts = tf.convert_to_tensor([e[0] for e in samples])
        # actions = tf.convert_to_tensor([e[1] for e in samples])
        # rewards = tf.convert_to_tensor([e[2] for e in samples], dtype='float32')
        # s_ts1 = tf.convert_to_tensor([e[3] for e in samples])
        # dones = tf.convert_to_tensor([e[4] for e in samples], dtype='float32')

        s_ts = tf.convert_to_tensor(s_ts)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        s_ts1 = tf.convert_to_tensor(s_ts1)
        dones = tf.convert_to_tensor(dones)
        return s_ts, actions, rewards, s_ts1, dones

    def update_err(self,q_value, q_target):       #maybe 
        err_batch = np.zeros((len(q_target), 1))
        for i in range(len(q_target)):
            err = 0
            for o in range(len(q_target[0])):
                err += np.abs(q_value[i][o] - q_target[i][o])
            err_batch[i][0] = err / len(q_target[0])
        return err_batch

    def save_model(self):
        try:
            self.target_actor_model.save(f"carla_ddpg_actor.h5", include_optimizer=False)
            self.target_critic_model.save(f"carla_ddpg_critic.h5", include_optimizer=False)
            self.target_critic_model_2.save(f"carla_ddpg_critic_2.h5", include_optimizer=False)
            print("------------------------------- save model ----------------------------------")
        except:
            print('---------------------------Can not save model----------------------------------')

    def load_model(self):
        try:
            self.actor_model=load_model(f"carla_ddpg_actor.h5")
            self.critic_model=load_model(f"carla_ddpg_critic.h5")
            self.critic_model_2=load_model(f"carla_ddpg_critic_2.h5")
            print("------------------------------- load model ----------------------------------")
        except:
            print('---------------------------Can not load model----------------------------------')

    def target_setweight(self):
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        self.target_critic_model_2.set_weights(self.critic_model_2.get_weights())
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        
def plot(episodes, rewards):
    if PLOT:
        # clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.title('episode. reward')
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Each Episode Reward')
        plt.savefig('reward_episode.png')
        plt.close('all')
        # plt.show()
    else: pass
if __name__ == "__main__":
    signal.signal(signal.SIGINT, my_handler)    #stop signal
    stop = False
    final_step = []
    rewards = []

    agent = Agent()
    agent.load_model()
    # adam_a = Adam(learning_rate=0.001)
    # adam_c = Adam(learning_rate=0.002)
    # agent.critic_model.compile(loss="mse",optimizer=adam_c, metrics=["acc"])
    # agent.critic_model_2.compile(loss="mse",optimizer=adam_c, metrics=["acc"])
    # agent.actor_model.compile(loss="mse",optimizer=adam_a, metrics=["acc"])
    # K.clear_session()
    agent.target_setweight()  
    print(agent.actor_model)
    env = CarEnv(SEGMENTATION, STATE_SIZE)

    t0 = time.time()
    # while stop is False:
    #     try:
    if TRAIN :
        for e in range(TRAIN_EPISODES):
            obs = env.reset()
            epoch_reward = 0
            step = 0
            s_t = np.reshape(obs, (1,STATE_SIZE,STATE_SIZE,3))
            for t in range(5000):
                # a_time = time.time()

                step += 1
                s_t = np.reshape(s_t, (1,STATE_SIZE,STATE_SIZE,3))
                # env.world.wait_for_tick()

                # control = env.agent_action()
                control = None
                get_action = np.reshape(env.vehicle.get_control().steer, (1,ACTION_DIM))
                a_predict, action = agent.act(s_t, control=control, get_action=get_action)
                action = np.reshape(action, (1,ACTION_DIM))

                obs_, reward, done, info = env.step(action[0])
                # img = np.reshape(s_t, (STATE_SIZE,STATE_SIZE,3))
                # cv2.putText(img, "%f"%action[0], (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,100,255),1,cv2.LINE_AA)
                # cv2.imshow("", img)
                # cv2.waitKey(1)

                s_t = np.reshape(s_t, (STATE_SIZE,STATE_SIZE,3))
                s_t1 = np.reshape(obs_, (STATE_SIZE,STATE_SIZE,3))
                reward = np.reshape(reward, (1))
                done = np.reshape(done, (1))
                
                agent.remember(s_t, action[0], reward, s_t1, done)
                epoch_reward +=reward[0]
    
                print("episode:{} step:{} predict_action:{} action:{} reward:{}".format(e,step,a_predict,action,reward))
            
                s_t = s_t1

                agent.train()
                # a_time = time.time()
                # print('==========================================time==================================', float(time.time()-a_time))

                if step % 3 ==0:
                    agent.update_target()

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
            if e == 0:
                    rewards.append(epoch_reward)
            else:
                rewards.append(rewards[-1] * 0.9 + epoch_reward * 0.1)
            final_step.append(step)

            wandb.log({"rewards": rewards[-1], "final_step": step})

            if e%2000 == 0 and e is not 0:
                plot(final_step,rewards)
        
            env.des()

            if e%50 == 0 and e is not 0:
            # if e%500 == 0 and e is not 0:
                agent.save_model()
            if step >= 2000 and e > 10:
                agent.save_model()
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        e + 1, TRAIN_EPISODES, epoch_reward, time.time() - t0))
    else:
        step_list = []
        reward_list = []
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            final_step = 0
            for step in range(5000):
                state = np.reshape(state, (1,STATE_SIZE,STATE_SIZE,3))

                get_action = np.reshape(env.vehicle.get_control().steer, (1,ACTION_DIM))
                action = agent.actor_model([state, get_action])
                # action = np.squeeze(action)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                final_step += 1
                # print("episode:{} step:{}  action:{} reward:{}".format(episode,step,action,reward))

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
            reward_list.append(episode_reward)
            step_list.append(step)
            print("the average of reward is", np.mean(reward_list))
            print("the average of step is", np.mean(step_list))
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Final_step {}| Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward, final_step,time.time() - t0))
        # except: print("===== Something is wrong =====")
                