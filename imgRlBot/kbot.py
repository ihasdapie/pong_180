import numpy as np
import os
import copy
import cv2
from utils import render_frame
from utils import gameData
from tensorflow import keras
import re
from po_NN_g import mdlmngr

#############
# Main block
SCALE_FACTOR = 5
TABLE_SIZE = (440, 280)
GAMMA = 0.995
MODEL_SHAPE = (TABLE_SIZE[0]//SCALE_FACTOR, TABLE_SIZE[1]//SCALE_FACTOR)
gd = gameData()

# load models...
mm = mdlmngr.from_scratch(MODEL_SHAPE)
#############


def update_reward(score):
    global gd
    if score[0] == gd.last_score[0] and score[1] == gd.last_score[1]:
        gd.reward = 0
    else:
        gd.reset = True
        if gd.cur_side == 'left':
            gd.cur_reward = score[0] - gd.last_score[0] + gd.last_score[1] - score[1]
        else:
            gd.cur_reward = score[1] - gd.last_score[1] + gd.last_score[0] - score[0]

def check_side(paddle_frect):
    global gd
    if paddle_frect.pos[0] < 100:
        side = 'left'
    else:
        side = 'right'
    if side != gd.cur_side:
        gd.cur_side = side
        gd.last_score = [0,0]

def kbot_pongbot(paddle_frect, other_paddle_frect, ball_frect, table_size, score = []):
    global gd, SCALE_FACTOR, mm, GAMMA
    check_side(paddle_frect)
    update_reward(score)
    
    half_paddle_width = paddle_frect.size[0]/2
    half_paddle_height = paddle_frect.size[1]/2
    half_ball_dim = ball_frect.size[0]/2

    cur_img = render_frame(paddle_frect, other_paddle_frect, ball_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim, SCALE_FACTOR)
    diff_img = cur_img - gd.xround[-1] if len(gd.xround) > 0 else cur_img # looking at the iamge difference tells us more

    move = mm.create_prediction(gd.cur_side, diff_img)

    y = 1.0 if move == 'up' else 0.0
    gd.yround.append(y)
    gd.xround.append(diff_img)
    gd.rround.append(gd.cur_reward)

    #update global variables
    gd.frame += 1
    gd.last_score = copy.deepcopy(score)

    if gd.reset:
        # I suppose this should be run on another thread so that we don't have to wait inbetween rounds
        # not saving game data right now... a problem for another day i suppose
        gd.xtrain.append(copy.deepcopy(gd.xround))
        gd.ytrain.append(copy.deepcopy(gd.yround))
        gd.rtrain.append(copy.deepcopy(gd.rround))
        gd.getsamples()
        
        # clear vars
        gd.frame = 1
        gd.cur_reward = 1
        gd.xround.clear()
        gd.yround.clear()
        gd.rround.clear()

        gd.reset = False        
        # run training on the correct bot 
    return move
