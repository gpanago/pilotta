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
    for j in range(4):
        for i in range(8):
            temp.append(cards[inc1 + i + 32*j])
    inc2 = 8*(s2 - 1)
    for j in range(4):
        for i in range(8):
            cards[inc1 + i + 32*j] = cards[inc2 + i + 32*j]
    for j in range(4):
        for i in range(8):
            cards[inc2 + i + 32*j] = temp[i + 8*j]

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

def load(files):
    x = []
    y = []
    hands = []
    trumps = []
    for i in files:
        with open(i) as f:
            for line in f:
                if "Trump suit" in line:
                    trump = int(line.split(':')[1])
                    trumps.append(trump)
                    elem = [0]*128
                elif "Player " in line:
                    hands.append(line)
                    player_z = "Player 0" in line
                    player_o = "Player 1" in line
                    player_t = "Player 2" in line
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
                        elif player_t:
                            elem[8*(s-1) + c - 7 + 32] = 1
                        elif player_o:
                            elem[8*(s-1) + c - 7 + 64] = 1
                        else:
                            elem[8*(s-1) + c - 7 + 96] = 1
                    if not player_z and not player_o and not player_t:
                        sort_hand(elem)
                        x.append(elem)
                elif "Score" in line:
                    y.append(int(line.split('.')[0].split(':')[1]))
    return (x, y, hands, trumps)
