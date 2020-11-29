import cv2
from collections import deque
import numpy as np
import os
def render_frame(p1_frect, p2_frect, b_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim, SCALE_FACTOR):
    img = np.zeros((table_size[1], table_size[0]))
    # draw in paddles in white
    top_left_x = round(p1_frect.pos[0]-half_paddle_width)
    top_left_y = round(p1_frect.pos[1]+half_paddle_height)
    bottom_right_x = round(p1_frect.pos[0] + half_paddle_width)
    bottom_right_y = round(p1_frect.pos[1] - half_paddle_height)
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 1, -1)

    top_left_x = round(p2_frect.pos[0]-half_paddle_width)
    top_left_y = round(p2_frect.pos[1]+half_paddle_height)
    bottom_right_x  = round(p2_frect.pos[0] + half_paddle_width)
    bottom_right_y = round(p2_frect.pos[1] - half_paddle_height)
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 1, -1)
    # draw in ball
    top_left_x = round(b_frect.pos[0]-half_ball_dim)
    top_left_y = round(b_frect.pos[1]+half_ball_dim)
    bottom_right_x = round(b_frect.pos[0] + half_ball_dim)
    bottom_right_y = round(b_frect.pos[1] - half_ball_dim)
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 1, -1)
    # downscale # can be downscaled more if necessary...
    return img[::SCALE_FACTOR, ::SCALE_FACTOR].transpose() # cv2 and what we use use different conventions
    # 2D array!

class gameData:
    def __init__(self):
        self.xtrain = deque()
        self.ytrain = deque()
        self.rtrain = deque()

        self.xround = deque()
        self.yround = deque()
        self.rround = deque()

        self.cur_side = 'left'
        self.last_score = [0,0]
        self.cur_reward = 0
        self.frame = 1
        self.reset=False

    def refresh(self):
        self.xtrain.clear()
        self.ytrain.clear()
        self.rtrain.clear()
        self.xround.clear()
        self.yround.clear()
        self.rround.clear()

        self.cur_side = 'left'
        self.last_score = [0,0]
        self.cur_reward = 0
        self.frame = 1
        self.reset=False

 
    def convenience_export(self):
        return (np.array(self.xround), np.array(self.yround), np.array(self.rtrain))
    
    def export_numpy_round(self):
        return (np.array(self.xround), np.array(self.yround), np.array(self.rround))
    def export_numpy_train(self):
        return np.array(self.xtrain), np.array(self.ytrain), np.array(self.rtrain)
    
    def getsamples(self):
        print("X----------")
        print(self.xround[-1])
        print("y----------")
        print(self.yround[-1])
        print("r-----------")
        print(self.rround[-1])


    def save_train(self):
        # lol rip this chokes on big matricis
        paths = os.listdir('./data/')
        if len(paths) == 0:
            n = 1
        else:
            n = max([re.findall("\d+", i)[0] for i in paths]) + 1
        # format = ./data/{type}_num.txt
        # can be loaded again with np.loadtxt...
        x, y, r = self.export_numpy_train()
        np.savetxt("./data/x_train_{n}".format(n=n), x)
        np.savetxt("./data/x_train_{n}".format(n=n), y)
        np.savetxt("./data/x_train_{n}".format(n=n), z)


