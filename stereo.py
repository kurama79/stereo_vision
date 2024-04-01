'''
    Tarea 7 - Stereo Matching
        En esta tarea se busca implementar un algoritmo de Stereovision, con la metrica de Census, junto con 
        Belief Propagation.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

from rectification import stereoRectification

# Funcion de la metricaCensus-Transform
def censusTransform(left_img, right_img, window_size, disparity):

    def census(img, window_size):

        same = np.int64(window_size/2)
        height, width = img.shape
        census = np.zeros((height-2*same, width-2*same), dtype=np.int64)

        cp = img[same:height-2, same:width-2]

        offsets = []
        for i in range(window_size):
            for j in range(window_size):
                if not j == same == i:
                    offsets.append((j, i))

        for u, v in offsets:
            census = (census << 1) | (img[v:v+height-2*same, u:u+width-2*same] >= cp)

        return census

    height, width, max_disparity = disparity.shape

    left_census = census(left_img[:,max_disparity:width], window_size)
    right_census = census(right_img[:,0:width-max_disparity], window_size)
    dispmap=np.zeros((left_census.shape[0], left_census.shape[1], max_disparity), dtype=np.int64)

    for i in range(left_census.shape[0]):
        for j in range(max_disparity, left_census.shape[1]):
            for k in range(max_disparity):
                dispmap[i,j,k] = '{:025b}'.format(((right_census[i,j-k]) ^ (left_census[i,j]))).count('1') # Formato Binario

    return dispmap

def sendMessages(disparity, rounds):

    def maxProduct(self_disp, neighbor_messages):

        max_disparity = len(self_disp)
        norm_const = 1
        next_message = [sys.maxsize for i in range(max_disparity)]

        for i in range(max_disparity):
            for j in range(max_disparity):

                contribution = 0
                for neighbor_message in neighbor_messages:
                    contribution += neighbor_message[j] # Sumatoria de los mensajes

                aux = self_disp[j] + penalityFunction(i, j) + contribution

                if aux < next_message[i]:
                    next_message[i] = aux

        next_message = norm_const*next_message / np.sum(next_message)

        return next_message

    # Funcion cuadratica
    def penalityFunction(self_disp, neighbor_disp, lambda_const=50):
        return lambda_const * (self_disp**2 + neighbor_disp**2)

    directions  = ['right', 'left', 'up', 'down']
    directions_size = len(directions)
    height, width, max_disparity = disparity.shape

    messages = np.ones((height, width, directions_size, max_disparity))
    next_messages = np.ones((height, width, directions_size, max_disparity))

    count = 0
    for round_n in range(rounds):
        for dir_n in directions:
            for i in range(height):
                for j in range(width):

                    neighbor_messages = []
                    for k in range(directions_size):
                        if directions[k] != dir_n:
                            neighbor_messages.append(messages[i,j,k])

                    try:
                        if dir_n == 'right':
                            next_messages[i,j+1,directions.index('left')] = maxProduct(disparity[i,j], neighbor_messages)

                        if dir_n == 'left':
                            next_messages[i,j-1,directions.index('right')] = maxProduct(disparity[i,j], neighbor_messages)

                        if dir_n == 'up':
                            next_messages[i+1,j,directions.index('down')] = maxProduct(disparity[i,j], neighbor_messages)

                        if dir_n == 'down':
                            next_messages[i-1,j,directions.index('up')] = maxProduct(disparity[i,j], neighbor_messages)

                    except (ValueError, IndexError) as e:
                        pass

        messages = next_messages

    return messages

def getFinalDisparity(disparity, messages):

    height, width, max_disparity = disparity.shape
    final_disparity = np.zeros((height, width))

    for i in range(height):
        for j in range(width):

            best_disparity = 0
            best_cost = sys.maxsize

            for k in range(max_disparity):

                neighbor_cost = 0
                for n in messages[i,j]:
                    neighbor_cost += n[k]

                if disparity[i,j,k] + neighbor_cost < best_cost:

                    best_cost = disparity[i,j,k] + neighbor_cost
                    best_disparity = k

                final_disparity[i,j] = best_disparity

    return final_disparity

# Implementando Algoritmo Belief Propagation
def stereo_beliefPropagation(left_img, right_img, max_disparity=10, window_size=5, rounds=3):

    height, width = left_img.shape
    disparity = np.zeros((height, width, max_disparity),dtype=np.int64)
    disparity1 = np.zeros((height, width, max_disparity),dtype=np.int64)

    disparity1 = censusTransform(left_img, right_img, window_size, disparity1)

    messages = sendMessages(disparity1, rounds)

    final_disparity = getFinalDisparity(disparity1, messages)

    return final_disparity

# Inicio del programa...
if __name__ == "__main__":

    # Leyendo la imagen
    img1 = cv2.imread('images/left01137.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/right01137.png', cv2.IMREAD_GRAYSCALE)

    img_rect1, img_rect2 = stereoRectification(img1, img2)

    disp = stereo_beliefPropagation(img_rect1, img_rect2)

    plt.imshow(disp, cmap=plt.get_cmap('gray'))
    plt.show()