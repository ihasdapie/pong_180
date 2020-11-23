import numpy as np
import cv2
import tensorflow as tf
import random
from collections import deque
from pygame.locals import Rect
TABLE_SIZE = (440, 280)
SCALE_FACTOR = 5
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others


def weight_layer(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

def bias_layer(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1]. padding="SAME")


def createNetwork(input_dim):
    global ACTIONS
    W_conv1 = weight_layer([8,8,4,32])
    b_conv1 = bias_layer([32])

    W_conv2 = weight_layer([4,4,32,64])
    b_conv2 = bias_layer([64])

    W_conv3 = weight_variable([3,3,64,64])
    b_conv2 = bias_layer([64])

    W_dense1 = weight_variable([1600,512])
    b_dense2 = bias_variable([512])

    W_dense2 = weight_variable([512, ACTIONS])
    b_dense2 = bias_variable([ACTIONS])

    s = tf.placeholder("float", [None, input_dim[0], input_dim[1], 4])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_dense1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_dense1) + b_dense1)

    # readout layer
    readout = tf.matmul(h_dense1, W_dense2) + b_dense2

    return s, readout, h_dense1


def cost_fnctn(readout, ):
    global ACTIONS
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y-readout_action))
    return cost


obs_history = deque()


def trainloop(s, readout, h_dense1, sess, img):
    global ACTIONS, obs_history, TABLE_SIZE, SCALE_FACTOR, ACTIONS, 
    global GAMMA, OBSERVE, EXPLORE, FINAL_EPSILON, INITIAL_EPSILON, REPLAY_MEMORY, BATCH, K

    cost = cost_fnctn(cost_fnctn)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    no_action = no.zeros(ACTIONS)
    do_nothing[0] = 1
    
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # run the selected action and observe next state and reward
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print "TIMESTEP", t, "/ STATE", state, "/ LINES", game_state.total_lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


class pongBoard:
    def __init__(self, board_size, p1_size, p1_loc, p2_size, p2_loc, ball_size, ball_loc)



# may need to add a rendered frame var if we have to schedule tasks slower than once per frame

prev_img = np.zeros(TABLE_SIZE)


def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    # reeeeeeeee
    global resume, render
    global RENDERED_FRAME, SCALE_FACTOR, H, batch_size, learning_rate, gamma, decay_rate, model
    global i, D, prev_img, xs, hs, dlogps, drs, running_reward, reward_sum, episode_number
    # Some of these variables should be calcuated only once and stored as a global
    half_paddle_width = paddle_frect.size[0]/2
    half_paddle_height = paddle_frect.size[1]/2
    half_ball_dim = ball_frect.size[0]/2
    cur_img = render_frame(paddle_frect, other_paddle_frect, ball_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim)
    # plt.imsave('graphics/1.png', img)
    diff_img = cur_img - prev_img if prev_img is not None else np.zeros(D)





def get_next_state()


def render_frame(p1_frect, p2_frect, b_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim):
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

class pongEmulator:
    def __init__(self, table_dim, margins, p1, p2, b):
        self.p1 = fRect(p1.pos, p2.size)
        self.p2 = fRect(p1.pos, p2.size)
        self.b = fRect(b.pos, b.size)
        self.dim = (table_dim[0], table_dim[1])
        self.walls_Rects = [Rect((-100, -100), (table_size[0]+200, 100)),
                       Rect((-100, table_size[1]), (table_size[0]+200, 100))]


    def update(self, p1, p2, b):
        # update pong emulator for when we can get new position data i.e. each time pong_ai is called
        self.p1.pos = p1.pos
        self.p2.pos = p2.pos
        self.b.pos = b.pos
    
    def move_ball(self, velocity): 
        inv_move_factor = int((velocity[0]**2+velocity[1]**2)**.5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                self.step_ball(velocity, 1./inv_move_factor)
        else:
            self.step_ball(velocity, 1)
        
    def step_ball(self, velocity, move_factor):
        #
        moved = 0
        for wall_rect in self.walls_Rects:
            if self.b.get_rect().colliderect(wall_rect):
                c = 0        
                while self.b.frect.get_rect().colliderect(wall_rect):
                    self.b.frect.move_ip(-.1*velocity[0], -.1*velocity[1], move_factor)
                    c += 1 # this basically tells us how far the ball has traveled into the wall
                # r1 = 1+2*(random.random()-.5)*self.dust_error
                # r2 = 1+2*(random.random()-.5)*self.dust_error
                    # r1 = 1 # I think this is correct? Because dust_error is 0
                   #  r2 = 1 
                nv = (self.speed[0], -1*self.speed[1]) # wallbounce, r1/r2 = 1
                while c > 0 or self.b.frect.get_rect().colliderect(wall_rect):
                    self.b.frect.move_ip(.1*velocity[0], .1*velocity[1], move_factor)
                    c -= 1 # move by roughly the same amount as the ball had traveled into the wall
                moved = 1

        for paddle in [self.p1, self.p2]:
            facing = 1 if paddle_frect.pos[0] > table_size[0]/2 else 0
            if self.b.frect.intersect(paddle):
                if (paddle.facing == 1 and self.get_center()[0] < paddle.frect.pos[0] + paddle.frect.size[0]/2) or \
                (paddle.facing == 0 and self.get_center()[0] > paddle.frect.pos[0] + paddle.frect.size[0]/2):
                    continue
                
                c = 0
                
                while self.frect.intersect(paddle.frect) and not self.frect.get_rect().colliderect(walls_Rects[0]) and not self.frect.get_rect().colliderect(walls_Rects[1]):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    
                    c += 1
                theta = paddle.get_angle(self.frect.pos[1]+.5*self.frect.size[1])
                

                v = self.speed

                v = [math.cos(theta)*v[0]-math.sin(theta)*v[1],
                             math.sin(theta)*v[0]+math.cos(theta)*v[1]]

                v[0] = -v[0]

                v = [math.cos(-theta)*v[0]-math.sin(-theta)*v[1],
                              math.cos(-theta)*v[1]+math.sin(-theta)*v[0]]


                # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
                if  v[0]*(2*paddle.facing-1) < 1: # ball is not traveling (a) away from paddle (b) at a sufficient speed
                    v[1] = (v[1]/abs(v[1]))*math.sqrt(v[0]**2 + v[1]**2 - 1) # transform y velocity so as to maintain the speed
                    v[0] = (2*paddle.facing-1) # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to the  increase by 1.2

                #a bit hacky, prevent multiple bounces from accelerating
                #the ball too much
                if not paddle is self.prev_bounce:
                    self.speed = (v[0]*self.paddle_bounce, v[1]*self.paddle_bounce)
                else:
                    self.speed = (v[0], v[1])
                self.prev_bounce = paddle
                

                while c > 0 or self.frect.intersect(paddle.frect):
                
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    
                    c -= 1
                
                moved = 1
                

        if not moved:
            self.frect.move_ip(self.speed[0], self.speed[1], move_factor)



class fRect:
    """Like PyGame's Rect class, but with floating point coordinates"""
    def __init__(self, pos, size):
        self.pos = (pos[0], pos[1])
        self.size = (size[0], size[1])
    def move(self, x, y):
        return fRect((self.pos[0]+x, self.pos[1]+y), self.size)

    def move_ip(self, x, y, move_factor = 1):
        self.pos = (self.pos[0] + x*move_factor, self.pos[1] + y*move_factor)

    def get_rect(self):
        return Rect(self.pos, self.size)

    def copy(self):
        return fRect(self.pos, self.size)

    def intersect(self, other_frect):
        # two rectangles intersect iff both x and y projections intersect
        for i in range(2):
            if self.pos[i] < other_frect.pos[i]: # projection of self begins to the left
                if other_frect.pos[i] >= self.pos[i] + self.size[i]:
                    return 0
            elif self.pos[i] > other_frect.pos[i]:
                if self.pos[i] >= other_frect.pos[i] + other_frect.size[i]:
                    return 0
        return 1 #self.size > 0 and other_frect.size > 0








