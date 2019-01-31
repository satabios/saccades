import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import random

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

screen_size = 84
speed = 1

screen = np.zeros((screen_size,screen_size))

def dir(x,y):

    if(x <= 0):
        if(y <=0):
            (X, Y) = random.sample([(x, y + speed), (x + speed, y + speed), (x + speed, y)], 1)[0]
        elif(y == screen_size):
            (X, Y) = random.sample([(x, y - speed), (x + speed, y), (x + speed, y - speed)], 1)[0]
        else:
            (X,Y) = random.sample([(x,y+speed),(x,y-speed),(x+speed,y),(x+speed,y+speed),(x+speed,y-speed)],1)[0]
    elif(x == screen_size):
        if(y == screen_size):
            (X, Y) = random.sample([(x, y - speed),(x-speed,y-speed),(x-speed,y) ], 1)[0]
        elif(y == 0):
            (X ,Y) = random.sample([(x-speed,y),(x-speed,y+speed),(x+speed,y+speed)],1)[0]
        else:
            (X,Y) = random.sample([(x,y-speed),(x,y+speed),(x-speed,y),(x-speed,y-speed),(x-speed,y+speed)],1)[0]

    else:
        (X,Y) = random.sample([(x,y+1),(x+1,y+1),(x+1,y),(x+1,y-1),(x,y-1),(x-1,y-1),(x-1,y),(x-1,y+1)],1)[0]


    return X,Y

if __name__ == '__main__':


    x = randint(0,screen_size-x_train.shape[1])
    y = randint(0,screen_size-x_train.shape[1])
    track =[]
    for pt in range(1000):
        screen = np.zeros((screen_size, screen_size))
        x,y = dir(x,y)
        screen[x:x+28, y:y+28] = x_train[0,:,:]
        plt.imshow(screen)
        track.append((x,y))

        plt.show()



