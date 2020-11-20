import numpy as np
import cv2
import pickle
TABLE_SIZE = (440, 280)
SCALE_FACTOR = 5
# may need to add a rendered frame var if we have to schedule tasks slower than once per frame


H = 200 # number of hidden layer neurons
batch_size = 10 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # learning rate used in RMS prop
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
resume = True # resume training from previous checkpoint (from save.p  file)?
render = False # render video output?

# model initialization
D = (TABLE_SIZE[0]//SCALE_FACTOR)*(TABLE_SIZE[1]//SCALE_FACTOR)

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
    model['W2'] = np.random.randn(H) / np.sqrt(H) # Shape will be H

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

prev_img = None
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0


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
    prev_img = cur_img
    

    # forward-prop
    aprob, h = policy_forward(diff_img)

    # commad convention: 2 = UP, 3 = Down, 0 = no movement
    action = 2 if np.random.uniform() < aprob else 3
    xs.append(diff_img)
    hs.append(h)

    y = 1 if action == 2 else 0 # fake label for NN (reinforcement learning)

    dlogpsa.append(y-aprob)



def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    """ this function discounts from the action closest to the end of the completed game backwards
    so that the most recent action has a greater weight """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)): 
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x) # (H x D) . (D x 1) = (H x 1) (200 x 1)
    h[h<0] = 0 # ReLU introduces non-linearity
    logp = np.dot(model['W2'], h) # This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
    p = sigmoid(logp)  # squashes output to  between 0 & 1 range
    return p, h # return probability of taking action 2 (UP), and hidden state

def policy_backward(eph, epx, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    """ Manual implementation of a backward prop"""
    """ It takes an array of the hidden states that corresponds to all the images that were
    fed to the NN (for the entire episode, so a bunch of games) and their corresponding logp"""
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

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


