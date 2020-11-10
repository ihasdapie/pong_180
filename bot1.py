import math

ball_pos_history = [(1,2), (3,4), (4,5)] # [(x, y), (x,y) ..
# just put some junk in there at first
predicted_pos = 133+7


def get_velocity(p1, p2):
    return ((p2[0]-p1[0], p2[1]-p1[1]))
def mag(tup):
    return (((tup[0]**2) +(tup[1]**2))**0.5)

def isclose(a, b, rel_tol=1e-03, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def get_velocity_flip(ph):
    # just need to find when it speeds up!
    # not reliable b.c. float calculations...
    # need to set threshold
    try:
        return not isclose(mag(get_velocity(ph[-1], ph[-2])), mag(get_velocity(ph[-2], ph[-3])))
    except:
        return False
    # for checking for sign flip
    # in retrospect i should pre-calculate get_velocity
    # return (get_velocity(ph[-1], ph[-2])[0] < 0 and get_velocity(ph[-2], ph[-3])[0] > 0) or (get_velocity(ph[-1], ph[-2])[0] > 0 and get_velocity(ph[-2], ph[-3])[0] < 0)

def if_flip(ph):
    return (get_velocity(ph[-1], ph[-2])[0] < 0 and get_velocity(ph[-2], ph[-3])[0] > 0) or (get_velocity(ph[-1], ph[-2])[0] > 0 and get_velocity(ph[-2], ph[-3])[0] < 0)

def predict_position(p1, p2, table_size):
    # returns distance between y = 0 and predicted final position when it "scores"
    #   /<-
    #  /
    # /
    #. p1
    # \
    #  \
    #   \-> p2

    v = get_velocity(p1, p2)
    #return (((table_size[0] - ((table_size[1]-p1[1])*(v[0]/v[1])))) % (table_size[1]*(v[0]/v[1])))*(v[1]/v[0])
    #Jack:

    #maybe change to something like:
    table_size = (table_size[0]-20, table_size[1]-30)
    try:
        v = list(v)
        v[0] = abs(v[0])
        #if no bounce
        if abs((p1[1]-7)*(v[0]/v[1])) > table_size[0]: return p1[1]+table_size[0]*(v[1]/v[0])

        #number of bounces
        n = (((table_size[0] - abs((p1[1]-7)*(v[0]/v[1])))) // (table_size[1]*(v[0]/v[1]))) + 1

        #cases
        if n%2 == 0 and v[1] > 0: return 7 + (((table_size[0] - abs((p1[1]-7)*(v[0]/v[1])))) % (table_size[1]*(v[0]/v[1])))*((v[1])/v[0])

        if n%2 == 0 and v[1] < 0: return 273 - (((table_size[0] - abs((p1[1]-7)*(v[0]/v[1])))) % (table_size[1]*(v[0]/v[1])))*((v[1])/v[0])

        if n%2 == 1 and v[1] > 0: return 273 - (((table_size[0] - abs((p1[1]-7)*(v[0]/v[1])))) % (table_size[1]*(v[0]/v[1])))*((v[1])/v[0])

        if n%2 == 1 and v[1] < 0: return 7 + (((table_size[0] - abs((p1[1]-7)*(v[0]/v[1])))) % (table_size[1]*(v[0]/v[1])))*((v[1])/v[0])

    except:
        return predicted_pos


def pongbot(paddle_frect, other_paddle_frect, ball_frect, table_size):
    global ball_pos_history # wish we had classes
    global predicted_pos
    '''return "up" or "down", depending on which way the paddle should go to
    align its centre with the centre of the ball, assuming the ball will
    not be moving

    Arguments:
    paddle_frect: a rectangle representing the coordinates of the paddle
                  paddle_frect.pos[0], paddle_frect.pos[1] is the top-left
                  corner of the rectangle.
                  paddle_frect.size[0], paddle_frect.size[1] are the dimensions
                  of the paddle along the x and y axis, respectively

    other_paddle_frect:
                  a rectangle representing the opponent paddle. It is formatted
                  in the same way as paddle_frect
    ball_frect:   a rectangle representing the ball. It is formatted in the
                  same way as paddle_frect
    table_size:   table_size[0], table_size[1] are the dimensions of the table,
                  along the x and the y axis respectively

    The coordinates look as follows:

     0             x
     |------------->
     |
     |
     |
 y   v
    '''
    ball_pos_history.append(ball_frect.pos)
    #print(ball_pos_history)
    v = get_velocity(ball_pos_history[-2], ball_pos_history[-1]) # -'ve x velocity means going to the left

    if if_flip(ball_pos_history): # could make slightly faster by calculating get_velocity in outside loop and passing v to func insteaad
        v = get_velocity(ball_pos_history[-2], ball_pos_history[-1]) # -'ve x velocity means going to the left
        # update predicted_pos only when opponent hits the ball
        if (((v[0] < 0) and (paddle_frect.pos[0] < table_size[0]/2)) or ((v[0]>0) and (paddle_frect.pos[0]>table_size[0]/2))):
            predicted_pos = predict_position(ball_pos_history[-2], ball_pos_history[-1], table_size)


    return controller(predicted_pos, paddle_frect.pos[1])

def controller(desired_pos, current_pos):
    # just something basic for now! move centroid of paddle to predicted pos
    if current_pos < desired_pos-35:
        return "down"
    else:
        return "up"
