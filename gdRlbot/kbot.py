import numpy as np
import os
import copy
import cv2
from utils import render_frame, gameData, predict_position
from tensorflow import keras
import re
from po_NN_g import mdlmngr
from collections import deque
#############
# Main block
SCALE_FACTOR = 8
TABLE_SIZE = (440, 280)
GAMMA = 0.995
# MODEL_SHAPE = (TABLE_SIZE[0]//SCALE_FACTOR, TABLE_SIZE[1]//SCALE_FACTOR)
FRAME_N = 3
MODEL_SHAPE = (10*FRAME_N, 1) # from size of frame...
gd = gameData()

# load models...
mm = mdlmngr.from_scratch(MODEL_SHAPE)
mdl_no = 26
# mm = mdlmngr.from_file(MODEL_SHAPE, 'mdls/r/r_{n}.h5'.format(n=mdl_no), \
                       # 'mdls/r/t_{n}.h5'.format(n=mdl_no), \
                       # 'mdls/l/r_{n}.h5'.format(n=mdl_no), \
                       # 'mdls/l/t_{n}.h5'.format(n=mdl_no))


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
    global gd, SCALE_FACTOR, mm, GAMMA, FRAME_N
    check_side(paddle_frect)
    update_reward(score)

    if len(gd.ball_pos_history) == 0: 
        # just fill some junk data in for the first 3 frames
        gd.ball_pos_history.append((0.0, 1.1))
        gd.ball_pos_history.append((1.0, 1.5))
        gd.ball_pos_history.append((2.0, 2.3))

    predicted_pos = predict_position(gd.ball_pos_history[-2], gd.ball_pos_history[-1], table_size, gd.ball_pos_history[-3][1])

    #update_frame_info
    cur_frame_x = deque()
    cur_frame_x.append(ball_frect.pos[0] / table_size[0] - 0.5)
    cur_frame_x.append(ball_frect.pos[1] / table_size[1] - 0.5)
    cur_frame_x.append(paddle_frect.pos[0] / table_size[0] - 0.5)
    cur_frame_x.append(paddle_frect.pos[1] / table_size[1] - 0.5)
    cur_frame_x.append((paddle_frect.pos[1]+70) / table_size[1] - 0.5)
    cur_frame_x.append(other_paddle_frect.pos[0] / table_size[0] - 0.5)
    cur_frame_x.append(other_paddle_frect.pos[1] / table_size[1] - 0.5)
    cur_frame_x.append((other_paddle_frect.pos[1]+70) / table_size[1] - 0.5)
    cur_frame_x.append(predicted_pos)


    # cur_frame_x is of dim. 10
    if len(cur_frame_x)//10 < FRAME_N:
        for _ in range(FRAME_N):
            gd.xround_cache.extend(cur_frame_x)
    gd.xround_cache.popleft()
    gd.xround_cache.extend(cur_frame_x)

    move = mm.create_prediction(gd.cur_side, cur_frame_x)
    
    y = 1.0 if move == 'up' else 0.0
    gd.yround.append(y)
    gd.xround.append(cur_frame_x)
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
