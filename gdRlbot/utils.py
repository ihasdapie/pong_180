import cv2
import re
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




# I'm assuming that this class is hogging a ton of memory
class gameData:
    def __init__(self):
        self.xtrain = deque()
        self.ytrain = deque()
        self.rtrain = deque()

        self.xround = deque()
        self.yround = deque()
        self.rround = deque()

        self.ball_pos_history = deque()

        self.xround_cache = deque()

        self.cur_side = 'left'
        self.last_score = [0,0]
        self.cur_reward = 0
        self.frame = 1
        self.reset=False

    def refresh(self):
        # it doesn't seam that re-declaring these variables
        # is clearing it from memory and nor is running deque().clear()
        # maybe try del?
        del self.xtrain, self.ytrain, self.rtrain, self.xround, self.yround, self.rround, self.ball_pos_history

        self.xtrain = deque()
        self.ytrain = deque()
        self.rtrain = deque()
        self.xround = deque()
        self.yround = deque()
        self.rround = deque()
        self.ball_pos_history = deque()

        self.cur_side = 'left'
        self.last_score = [0,0]
        self.cur_reward = 0
        self.frame = 1
        self.reset=False
 
    
    def export_numpy_round(self):
        return (np.array(self.xround), np.array(self.yround), np.array(self.rround))
    def export_numpy_train(self):
        return np.array(self.xtrain), np.array(self.ytrain), np.array(self.rtrain)
    
    def getsamples(self):
        print("--SAMPLES FROM MOST RECENT ROUND--")
        print("X----------")
        # print("LEN: ", len(self.xtrain), self.xtrain[-1])
        print("LEN: ", len(self.xtrain))
        print("y----------")
        # print("LEN: ", len(self.ytrain), self.ytrain[-1])
        print("LEN: ", len(self.ytrain))
        print("r-----------")
        # print("LEN: ", len(self.rtrain), self.rtrain[-1])
        print("LEN: ", len(self.rtrain))


    def save_train_data(self):
        # lol rip this chokes on big matricis
        paths = os.listdir('./data/')
        if len(paths) == 0:
            n = 1
        else:
            n = max([int(re.findall("\d+", i)[0]) for i in paths]) + 1
        # format = ./data/{type}_num.txt
        # can be loaded again with np.loadtxt...
        x, y, r = self.export_numpy_train()
        np.save("./data/x_train_{n}".format(n=n), x)
        np.save("./data/y_train_{n}".format(n=n), y)
        np.save("./data/r_train_{n}".format(n=n), r)

    def load_train_data(self):
        paths = os.listdir('./data/')
        if len(paths) == 0:
            n = 1
        else:
            n = max([int(re.findall("\d+", i)[0]) for i in paths])
        # format = ./data/{type}_num.txt
        # can be loaded again with np.loadtxt..
        self.xtrain = np.load("./data/x_train_{n}.npy".format(n=n) , allow_pickle=True)
        self.ytrain = np.load("./data/y_train_{n}.npy".format(n=n), allow_pickle=True )
        self.rtrain = np.load("./data/r_train_{n}.npy".format(n=n) , allow_pickle=True)

def get_velocity(p1, p2):
    return ((p2[0]-p1[0], p2[1]-p1[1]))


def predict_position(p1, p2, table_size, h):
    # returns distance between y = 0 and predicted final position when it "scores"
    #   /<-
    #  /
    # /
    #. p1
    # \
    #  \
    #   \-> p2
    #print("predictied height:", h)
    v = get_velocity(p1, p2)

    table_size = (table_size[0]-70, table_size[1]-15)
    try:
        v = list(v)
        v[0] = abs(v[0])
        #if no bounce
        if v[1] < 0 and abs(table_size[0]*(v[1]/v[0])) < p1[1]:
            return p1[1]+7.5+table_size[0]*(v[1]/v[0])

        if v[1] > 0 and abs(table_size[0]*(v[1]/v[0])) < table_size[1] - p1[1]:
            return p1[1]+7.5+table_size[0]*(v[1]/v[0])

        d1 = v[0]/abs(v[1])
        #number of bounces
        if v[1] > 0:
            d1 = (table_size[1] - p1[1])*d1
        else:
            d1 = p1[1]*d1

        n = (table_size[0] - d1) // (table_size[1]*(v[0]/abs(v[1]))) + 1

        a = (table_size[0] - d1) % (table_size[1]*(v[0]/abs(v[1])))

        #cases
        if n%2 == 0 and v[1] > 0:
            #k = 7.5 + a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 1:", k)
            #cache["case"] = "case 1"
            return 7.5 + a*(abs(v[1])/v[0])

        if n%2 == 0 and v[1] < 0:
            #k = 272.5 - a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 2:", k)
            #cache["case"] = "case 2"
            return 272.5 - a*(abs(v[1])/v[0])

        if n%2 == 1 and v[1] > 0:
            #k = 272.5 - a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 3:", k)
            #cache["case"] = "case 3"
            return 272.5 - a*(abs(v[1])/v[0])

        if n%2 == 1 and v[1] < 0:
            #k = 7.5 + a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 4:", k)
            #cache["case"] = "case 4"
            return 7.5 + a*(abs(v[1])/v[0])

    except:
        print("exception")
        return predicted_pos



