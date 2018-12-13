#!/usr/bin/python2

# Milestone model was
# model = keras.Sequential([
#    keras.layers.Dense(16, activation=tf.nn.relu,
#                       input_shape=(64,)),
#    keras.layers.Dense(16, activation=tf.nn.relu),
#    keras.layers.Dense(1, activation=tf.nn.sigmoid)])


import tensorflow as tf
from tensorflow import keras
import numpy as np

card_values = {0 : 0.02, 1 : 0.03, 2 : 0.04, 3 : 10, 4 : 2, 5 : 3, 6 : 4, 7 : 11}

def suit_value(cards, s):
    ret = 0
    inc = 8*(s-1)
    for i in range(8):
        if cards[inc + i] or cards[inc + i + 32]:
            ret += card_values[i]
    return ret

def swap_suits(cards, s1, s2):
    temp = []
    inc1 = 8*(s1 - 1)
    for i in range(8):
        temp.append(cards[inc1 + i])
    for i in range(8):
        temp.append(cards[inc1 + i + 32])
    inc2 = 8*(s2 - 1)
    for i in range(8):
        cards[inc1 + i] = cards[inc2 + i]
        cards[inc1 + i + 32] = cards[inc2 + i + 32]
    for i in range(8):
        cards[inc2 + i] = temp[i]
        cards[inc2 + i + 32] = temp[i + 8]

def sort_hand(cards):
    suit2 = suit_value(cards, 2)
    suit3 = suit_value(cards, 3)
    suit4 = suit_value(cards, 4)
    if suit2 < suit3:
        swap_suits(cards, 2, 3)
        suit2, suit3 = suit3, suit2
    if suit3 < suit4:
        swap_suits(cards, 3, 4)
        suit3, suit4 = suit4, suit3
    if suit2 < suit3:
        swap_suits(cards, 2, 3)
        suit2, suit3 = suit3, suit2
    
x = []
y = []
sigma = []
hands = []
trumps = []

for i in range(1,4):
    with open("data/data" + str(i)) as f:
        readhand = True
        for line in f:
            if "Data number" in line:
                elem = [0]*64
                score = []
                i = 0
                readhand = True
            elif readhand and "Trump suit" in line:
                trump = int(line.split(':')[1])
                trumps.append(trump)
            elif readhand and ("Player 0" in line or
                               "Player 2" in line):
                hands.append(line)
                player_z = "Player 0" in line
                for card in line.split(':')[1].split(' ')[1:]:
                    c = card.split(',')[0]
                    s = int(card.split(',')[1])
                    # make the trump suit always the 1 suit
                    if s == trump:
                        s = 1
                    elif s == 1:
                        s = trump
                    if c == 'J':
                        c = 11
                    elif c == 'Q':
                        c = 12
                    elif c == 'K':
                        c = 13
                    elif c == 'A':
                        c = 14
                    else:
                        c = int(c)
                    if player_z:
                        elem[8*(s-1) + c - 7] = 1
                    else:
                        elem[8*(s-1) + c - 7 + 32] = 1
                if not player_z:
                    readhand = False
                    sort_hand(elem)
                    x.append(elem)
            elif "Score" in line:
                score.append(int(line.split('.')[0].split(':')[1]))
                i += 1
                if i == 20:
                    mean = sum(score)/20
                    y.append(mean)
                    mysigma = 0
                    for j in score:
                        mysigma += (j - mean)**2
                    mysigma /= 20
                    sigma.append(np.sqrt(mysigma))
                    


print(len(x))
print(len(y))

y = np.array(y) / 162.
x = np.array(x)
train_data = x[:-2000]
train_labels = y[:-2000]
val_data = x[len(x)-2000:-1000]
val_labels = y[len(y)-2000:-1000]
test_data = x[len(x)-1000:]
test_labels = y[len(y)-1000:]


model = keras.Sequential([
    keras.layers.Dense(8, activation=tf.nn.elu,
                       input_shape=(64,)),
    keras.layers.Dense(8, activation=tf.nn.elu),
    keras.layers.Dense(8, activation=tf.nn.elu),
    keras.layers.Dense(8, activation=tf.nn.elu),
    keras.layers.Dense(8, activation=tf.nn.elu),
    keras.layers.Dense(8, activation=tf.nn.elu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0002),
              loss='mse',
              metrics=['mae'])

model.summary()

model.fit(train_data, train_labels, epochs=400)

val_loss, val_acc = model.evaluate(val_data, val_labels)
print('Val accuracy:', val_acc, val_acc*162)
print('Sigma:', sum(sigma[:10000])/10000)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc, test_acc*162)
