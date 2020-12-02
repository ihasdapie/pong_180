###################
# Pong + Reinforcement Learning + CNNs
###################
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
import re
import os
from collections import deque
from copy import deepcopy
tf.config.experimental_run_functions_eagerly(True)

# def loss(logit, label, reward, m):
#     entr = label * -tf.log(logit) + (1-label) * -tf.log(1-logit)
# return -tf.reduce_sum(reward * entr)
# logit is pred_y, label is y_true, reward is reward...
# l = y_true * log(y_pred)  + (1-y_true) * - log(1-y_pred)

def modified_jack_loss(eps_reward):
    print(eps_reward)
    # I think this might need to be gone over with because it reports a loss of 0 at a reward of 0... 
    def loss(y_true, y_pred):
        # prune pred b.c. of possible invalid nums (domain of log)
        pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.02,0.98))(y_pred)
        # tmp_loss = keras.layers.Lambda(lambda x: -y_true*keras.backend.log(x) + (y_true-1) * keras.backend.log(1-x))(pred)
        tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(pred)
        # tmp. loss is pos.
        # rewards has range -1, 1
        # 1 being good
        # want to 
        policy_loss=keras.layers.Multiply()([tmp_loss,eps_reward])
        # policy_loss = keras.backend.sum(policy_loss)
        return policy_loss
    return loss



# 0


# def modified_jack_loss(eps_reward):
#     # I think this might need to be gone over with because it reports a loss of 0 at a reward of 0... 
#     def loss(y_true, y_pred):
#         # prune pred b.c. of possible invalid nums (domain of log)
#         #     entr = label * -tf.log(logit) + (1-label) * -tf.log(1-logit)
#         pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.02,0.98))(y_pred)
#         tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(pred)
#         policy_loss=keras.layers.Multiply()([tmp_loss,eps_reward])
#         return policy_loss
#     return loss


def make_models(input_shape):
    # assuming SCALE_FACTOR = 5 
    # (88 x 56 x 1) -> conv2d -> conv2d -> flatten -> 1x1
    # not using Sequential gives a little more flexibility
    input_layer = keras.layers.Input(shape=input_shape)
    input_layer_plus_channel = keras.layers.Reshape((*(input_shape), 1))(input_layer)
    #------- Can modify model pretty easily here!
    conv1_layer = keras.layers.Conv2D(4, 8, activation='relu', strides=(3,3), padding='valid', use_bias=True, )(input_layer_plus_channel)
    maxpool1_layer = keras.layers.MaxPool2D(pool_size=(2,2))(conv1_layer) # I'd imagine there is a lot of redundancy in the frames...
    conv2_layer = keras.layers.Conv2D(8, 4, activation='relu', strides=(1,1), padding='valid', use_bias=True, )(maxpool1_layer) # Maybe just one CNN is enough
    flatten1_layer = keras.layers.Flatten()(conv2_layer)
    #--------
    output_layer = keras.layers.Dense(1, activation="sigmoid", use_bias=True)(flatten1_layer) # To Bias or Not To Bias?
    
    reward_layer = keras.layers.Input(shape=(1,), name='reward_layer')

    run_model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    train_model = keras.models.Model(inputs=input_layer, outputs=output_layer) 
    
    # train_model.compile(optimizer='adam', loss=modified_jack_loss(reward_layer))
    train_model.compile(optimizer='adam', loss='binary_crossentropy')
   
    return train_model, run_model

def convert_advantage_factor(r_train, gamma):
    # takes in r_train (list of lists), calculates adv_factor
    # flattens list and then normalizes
    print("rtrain raw----------------")
    # print(r_train)
    flatten = lambda t: [item for sublist in t for item in sublist]
    r_train_modified = []
    for r in r_train:
        tmp = []
        fac = r[-1]
        rlen = len(r)
        for i in range(rlen): 
            tmp.append(fac*gamma**(rlen-i))
        r_train_modified.append(tmp)
    
    r_train_modified = np.array(flatten(r_train_modified))
    # normalize
    r_train_modified -= np.mean(r_train_modified)
    r_train_modified /= np.std(r_train_modified)
    print("--------converted-advantage-factor-------")
    # print(r_train_modified)
    return r_train_modified


# def convert_advantage_factor(r_train, gamma):
#     # takes in r_train (list of lists), calculates adv_factor
#     # flattens list and then normalizes

#     # loss taken from  https://github.com/thinkingparticle/deep_rl_pong_keras/blob/master/reinforcement_learning_pong_keras_policy_gradients.ipynb

#     flatten = lambda t: [item for sublist in t for item in sublist]
#     r_train_modified = []
#     tmp_r = 0
#     for rd in r_train:
#         for i in range(len(rd)-1, -1, -1):
#             if rd[i] == 0:
#                 tmp_r = tmp_r * (1-gamma)
#                 r_train_modified.append(tmp_r)
#             else:
#                 tmp_r = rd[i]
#                 r_train_modified.append(tmp_r)

#     r_train_modified = np.array(r_train_modified)
#     # normalize
#     # r_train_modified -= np.mean(r_train_modified)
#     # r_train_modified /= np.std(r_train_modified)
#     print("--------converted-advantage-factor-------")
#     return np.flip(r_train_modified, 0)

class mdlmngr:
    # a messy class to manage dealing with models & functions defined in po_NN_g
    def __init__(self, right_run_model, right_train_model, left_run_model, left_train_model): 
        self.right_run_model = right_run_model
        self.right_train_model = right_train_model
        self.left_run_model = left_run_model
        self.left_train_model = left_train_model

    @classmethod
    def from_scratch(cls, input_shape):
        ltm, lrm = make_models(input_shape)
        rtm, rrm = make_models(input_shape)
        return cls(rrm, rtm, lrm, ltm)

    @classmethod
    def from_file(cls, right_run_model, right_train_model, left_run_model, left_train_model): 
        # pass relative path strings
        rrm = keras.models.load_model(right_run_model)
        rtm = keras.models.load_model(right_train_model)
        lrm = keras.models.load_model(left_train_model)
        ltm = keras.models.load_model(left_run_model)
        return cls(rrm, rtm, lrm, ltm)

    def save_models(self):
        # assuming path: ./mdls/l & ./mdls/r for left & right models, respectively. And left and right are of same num.
        # naming convention: {mdl_type}_num.h5
        paths = os.listdir('./mdls/l')
        try:
            paths.remove('.ipynb_checkpoint')
        except:
            pass
        if len(paths) == 0:
            n = 1
        else:
            n = max([int(re.findall("\d+", i)[0]) for i in paths]) + 1
        self.left_run_model.save('./mdls/l/r_{n}.h5'.format(n=str(n)))
        self.right_run_model.save('./mdls/r/r_{n}.h5'.format(n=str(n)))
        self.left_train_model.save('./mdls/l/t_{n}.h5'.format(n=str(n)))
        self.right_train_model.save('./mdls/r/t_{n}.h5'.format(n=str(n)))



    def train_models(self, side, x_train, y_train, r_train, gamma):
        # take from pongbot, np.arrays
        # x, r are training values, r_train must be computed each time (convert advantage factor)
        
        flatten = lambda t: [item for sublist in t for item in sublist]
        print("-------STARTING TRAINING------")
        # print(r_train)
        r_train  = convert_advantage_factor(r_train, gamma)
        r_train = np.expand_dims(r_train, 1)
        y_train = np.array(flatten(y_train))
        y_train = np.expand_dims(y_train, 1)
        x_train = np.array(flatten(x_train))

        # why are these ragged?

        print("-----SHAPES-------")
        print("X:", x_train.shape)
        print("Y:", y_train.shape)
        print("R:", r_train.shape)
        # print(type(x_train), type(x_train), type(r_train))
        # print(type(x_train[0]), type(x_train[0]), type(r_train[0]))
        print(r_train)

        if side == 'right':
            self.right_train_model.fit(x=x_train , y=y_train, \
                batch_size = 8, epochs=4, verbose=1, \
                validation_split = 0.1, shuffle=True, sample_weight = r_train)
        else:
            self.left_train_model.fit(x=x_train, y=y_train, \
                batch_size = 8, epochs=4, verbose=1, \
                validation_split = 0.1, shuffle=True, sample_weight = r_train)

    def create_prediction(self, side, x):
        if side == 'right':
            action_prob = self.right_run_model.predict(np.expand_dims(x, axis=0), batch_size=1)[0][0]
        else:
            action_prob = self.left_run_model.predict(np.expand_dims(x, axis=0), batch_size=1)[0][0]

        action = np.random.choice(a=[2,3],size=1,p=[action_prob, 1-action_prob])
        ret = 'up' if action == 2 else 'down'
        return ret


