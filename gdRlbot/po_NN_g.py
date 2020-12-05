###################
# Pong + Reinforcement Learning + CNNs
###################
import math
import os
import re
from collections import deque
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.config.experimental_run_functions_eagerly(True)

# def loss(logit, label, reward, m):
#     entr = label * -tf.log(logit) + (1-label) * -tf.log(1-logit)
# return -tf.reduce_sum(reward * entr)

def modified_jack_loss(eps_reward):
    def loss(y_true, y_pred):
        # prune pred b.c. of possible invalid nums (domain of log)
        pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.001,0.999))(y_pred)
        # tmp_loss = keras.layers.Lambda(lambda x: -y_true*keras.backend.log(x) + (y_true-1) * keras.backend.log(1-x))(pred)
        tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(pred)
        # tf.print(eps_reward) # doesn't work because tf print can't print tensors???
        policy_loss=keras.layers.Multiply()([tmp_loss,eps_reward])
        # why does this turn it into 0?
        return policy_loss
    return loss


# def modified_jack_loss(reward):
#   def custom_loss(y_true, y_pred):
#     return keras.backend.mean(keras.backend.square(y_pred - y_true) - keras.backend.square(y_true - reward), axis=-1)

#   return custom_loss
# # 0


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
    # shape = (n_item, 10*FRAME_N, 1)
    input_layer = keras.layers.Input(shape=input_shape)
    #------- Can modify model pretty easily here!
    fc1= keras.layers.Dense(units=input_shape[0], activation='relu', use_bias=True,)(input_layer)
    # fc2= keras.layers.Dense(units=50, activation='relu', use_bias=True,)(fc1)
    # fc3= keras.layers.Dense(units=100, activation='relu', use_bias=True,)(fc2)
    output_layer = keras.layers.Dense(1, activation="sigmoid", use_bias=True)(fc1)
    reward_layer = keras.layers.Input(shape=(1,), name='reward_layer')

    run_model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    train_model = keras.models.Model(inputs=[input_layer, reward_layer], outputs=output_layer) 
    # train_model.compile(optimizer='adam', loss=modified_jack_loss(reward_layer))
    train_model.compile(optimizer='adam', loss=modified_jack_loss(reward_layer))
    train_model.summary() # Why is this giving me an output where the layers are connected via [0][0] slices?
    run_model.summary()

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
    # print("--------converted-advantage-factor-------")
    # print(r_train_modified)
    return r_train_modified


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
    def from_file(cls, input_shape, right_run_model, right_train_model, left_run_model, left_train_model): 
        # get input shape
        # pass relative path strings
        ltm, lrm = make_models(input_shape)
        rtm, rrm = make_models(input_shape)
        ltm.load_weights(left_train_model)
        lrm.load_weights(left_run_model)
        rtm.load_weights(right_train_model)
        rrm.load_weights(right_run_model)
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


        print("-----SHAPES-------")
        print("X:", x_train.shape)
        print("Y:", y_train.shape)
        print("R:", r_train.shape)
        # print(type(x_train), type(x_train), type(r_train))
        # print(type(x_train[0]), type(x_train[0]), type(r_train[0]))
        # print(r_train)

        if side == 'right':
            self.right_train_model.fit(x=[x_train, r_train], y=y_train, \
                batch_size = 8, epochs=2, verbose=1, \
                validation_split = 0.1, shuffle=True ) # sampleweight = r_train 
        else:
            self.left_train_model.fit(x=[x_train, r_train], y=y_train, \
                batch_size = 8, epochs=2, verbose=1, \
                validation_split = 0.1, shuffle=True )
        keras.backend.clear_session()





    def create_prediction(self, side, x):
        if side == 'right':
            action_prob = self.right_run_model.predict(np.expand_dims(x, axis=0), batch_size=1)[0][0][0]
        else:
            action_prob = self.left_run_model.predict(np.expand_dims(x, axis=0), batch_size=1)[0][0][0]

        # print(action_prob, "-------------")
        action = np.random.choice(a=[2,3],size=1,p=[action_prob, 1-action_prob])
        ret = 'up' if action == 2 else 'down'
        return ret
