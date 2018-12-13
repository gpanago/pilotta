#!/usr/bin/python2

import tensorflow as tf
import numpy as np
import random
import sys
from collections import deque

import utils

# Change to True to train
training = True
# Change to False to continuously print an average score
display = False



no_a = 37
no_realizations = 20
possible_actions = np.zeros((no_a,no_a))
np.fill_diagonal(possible_actions, 1)
max_steps = 9
e_start = 1.0
e_end = 0.01
decay_rate = 0.00005
learning_rate = 0.0002
total_episodes = 10000000
if training:
    memory_size = 100000
else:
    memory_size = 10
batch_size = 64
pretrain_length = memory_size
gamma = 0.99
max_tau = 10000




tf.reset_default_graph()
evaluate = tf.keras.models.load_model('models/eval.h5')

class Pilotta:
    def __init__(self):
        self.deck = range(32)
        
    def set_hand(self, h, d):
        for i in d[:8]:
            h[i] = 1
        for i in d[8:16]:
            h[i+32] = 1
        
    def new_episode(self):
        random.shuffle(self.deck)
        self.hand = [0]*64
        self.prevbids = [0]*no_a
        self.set_hand(self.hand, self.deck)
        self.firstplays = True
        self.done = False
        self.nobids = 0
        self.maxbid = -1
        
    def get_state(self):
        if self.firstplays:
            return np.array(self.hand[:32] + self.prevbids)
        return np.array(self.hand[32:] + self.prevbids)

    def make_action(self, action):
        ind = np.argmax(action)
        self.nobids += 1
        self.firstplays = not self.firstplays
        
        # A pass
        if ind == 0:
            # The very first bid is a pass and the bidding continues
            if self.nobids == 1:
                self.prevbids[ind] = 1
                return 0
            # Both players passed and the hand is tossed
            if self.nobids == 2 and self.prevbids[0] == 1:
                self.done = True
                return -100

            self.done = True
            return self.evaluate()

        # It is illegal to bid less or the same as before
        if self.maxbid > 0 and ind < self.maxbid + 4 - ((self.maxbid - 1) % 4):
            self.done = True
            return -100

        self.maxbid = ind
        self.prevbids[ind] = 1
        return 0

    def is_episode_finished(self):
        return self.done

    def evaluate(self):
        trump = (self.maxbid - 1) % 4 + 1
        bid = (self.maxbid - trump) // 4 + 8
        opp_deck = self.deck[16:]
        decks = np.zeros((no_realizations, 128))
        for i in range(no_realizations):
            random.shuffle(opp_deck)
            for j in range(64):
                decks[i, j] = self.hand[j]
            for j in opp_deck[:8]:
                decks[i, j + 64] = 1
            for j in opp_deck[8:16]:
                decks[i, j + 96] = 1


        for i in range(no_realizations):
            if trump != 1:
                utils.swap_suits(decks[i, :], 1, trump)
            utils.sort_hand(decks[i, :])
            
        scores = evaluate.predict(decks) * 162
        score = 0
        for s in scores:
            if s < bid*10:
                score -= bid + 16
            else:
                s = int(s)
                r = s % 10
                if r >= 6:
                    s += 10
                s -= r
                s //= 10
                score += bid + s - (16 - s)
                
        score = float(score) / no_realizations
        return score

    def print_action(self, action):
        ind = np.argmax(action)
        if not self.firstplays:
            sys.stdout.write("Player A says: ")
        else:
            sys.stdout.write("Player B says: ")
        if ind == 0:
            print("Pass")
        else:
            trump = (ind - 1) % 4 + 1
            bid = (ind - trump) // 4 + 8
            print("{}".format(bid), ",{}".format(trump))

    def print_cards(self):
        sys.stdout.write("Player A has:")
        for c in sorted(self.deck[:8]):
            suit = c // 8 + 1
            card = (c % 8) + 7
            if card == 11:
                card = 'J'
            elif card == 12:
                card = 'Q'
            elif card == 13:
                card = 'K'
            elif card == 14:
                card = 'A'
            else:
                card = str(card)
            sys.stdout.write(' ')
            sys.stdout.write(card)
            sys.stdout.write(',')
            sys.stdout.write(str(suit))
        sys.stdout.write('\nPlayer B has:')
        for c in sorted(self.deck[8:16]):
            suit = c // 8 + 1
            card = (c % 8) + 7
            if card == 11:
                card = 'J'
            elif card == 12:
                card = 'Q'
            elif card == 13:
                card = 'K'
            elif card == 14:
                card = 'A'
            else:
                card = str(card)
            sys.stdout.write(' ')
            sys.stdout.write(card)
            sys.stdout.write(',')
            sys.stdout.write(str(suit))
        sys.stdout.write('\n')
    

class DDDQNNet:
    def __init__(self, learning_rate, name):
        self.learning_rate = learning_rate
        self.name = name

        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, 32 + no_a])
            self.actions_ = tf.placeholder(tf.float32, [None, no_a])
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1])
            self.targetQ = tf.placeholder(tf.float32, [None])
            
            self.layer1 = tf.layers.dense(self.inputs_, 32, activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.layer2 = tf.layers.dense(self.layer1, 32, activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.layer3 = tf.layers.dense(self.layer2, 32, activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.value_fc = tf.layers.dense(self.layer3, 32, activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.value = tf.layers.dense(self.value_fc, 1, activation = None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.advantage_fc = tf.layers.dense(self.layer3, 32, activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.advantage = tf.layers.dense(self.advantage_fc, no_a, activation = None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            self.absolute_errors = tf.abs(self.targetQ - self.Q)# for updating Sumtree
            
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.targetQ, self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


            
class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)
        
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

        
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

# Instantiate the two DQNetwork for the two players
DQNetwork = DDDQNNet(learning_rate, "DQNetwork")
TargetNetwork = DDDQNNet(learning_rate, "TargetNetwork")
DQNetwork2 = DDDQNNet(learning_rate, "DQNetwork2")
TargetNetwork2 = DDDQNNet(learning_rate, "TargetNetwork2")



# Instantiate memory
memory = Memory(memory_size)
memory2 = Memory(memory_size)
game = Pilotta()

# Render the environment
game.new_episode()

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = game.get_state()
    
    # Random action
    action = random.choice(possible_actions)
    
    # Get the rewards
    reward = game.make_action(action)
    
    # Look if the episode is finished
    done = game.is_episode_finished()

    if not done:
        reward += game.make_action(random.choice(possible_actions))
        done = game.is_episode_finished()
    
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        memory.store((state, action, reward, next_state, done))
        
        # Start a new episode
        game.new_episode()
        
        # First we need a state
        state = game.get_state()
        
    else:
        # Get the next state
        next_state = game.get_state()
        
        # Add experience to memory
        memory.store((state, action, reward, next_state, done))
        
        # Our state is now the next_state
        state = next_state

game.new_episode()
game.make_action(random.choice(possible_actions))

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = game.get_state()
    
    # Random action
    action = random.choice(possible_actions)
    
    # Get the rewards
    reward = game.make_action(action)
    
    # Look if the episode is finished
    done = game.is_episode_finished()

    if not done:
        reward += game.make_action(random.choice(possible_actions))
        done = game.is_episode_finished()
    
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        memory2.store((state, action, reward, next_state, done))
        
        # Start a new episode
        game.new_episode()
        game.make_action(random.choice(possible_actions))
        
        # First we need a state
        state = game.get_state()
        
    else:
        # Get the next state
        next_state = game.get_state()
        
        # Add experience to memory
        memory2.store((state, action, reward, next_state, done))
        
        # Our state is now the next_state
        state = next_state

firstp = True

def predict_action(decay_step, state, firstplayer):
    ran = np.random.rand()
    explore_prob = e_end + (e_start - e_end) * np.exp(-decay_rate * decay_step)
    if explore_prob > ran:
        action = random.choice(possible_actions)
    else:
        if firstplayer:
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1,) + state.shape)})
        else:
            Qs = sess.run(DQNetwork2.output, feed_dict = {DQNetwork2.inputs_: state.reshape((1,) + state.shape)})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_prob

# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():

    if firstp:
        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
        
        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
    else:
        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork2")
        
        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork2")
        

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
        
saver = tf.train.Saver()

if training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        evaluate = tf.keras.models.load_model('models/eval.h5')
        decay_step = 0
        tau = 0
        loss = 0
        
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        
        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            game.new_episode()
            state = game.get_state()
            if not firstp:
                action2, _ = predict_action(decay_step, state, True)
                game.make_action(action2)
                state = game.get_state()

            while step < max_steps//2:
                step += 1
                decay_step += 1
                tau += 1

                action, explore_probability = predict_action(decay_step, state, firstp)

                reward = game.make_action(action)
                done = game.is_episode_finished()
                if not done:
                    inter_state = game.get_state()
                    action2, _ = predict_action(decay_step, state, not firstp)
                    reward += game.make_action(action2)
                    done = game.is_episode_finished()
                episode_rewards.append(reward)
                        

                if done:
                    next_state = np.zeros(32 + no_a, dtype=np.int)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)

                    if episode % 20 == 0:
                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    if firstp:
                        memory.store((state, action, reward, next_state, done))
                    else:
                        memory2.store((state, action, reward, next_state, done))

                else:
                    next_state = game.get_state()
                    if firstp:
                        memory.store((state, action, reward, next_state, done))
                    else:
                        memory2.store((state, action, reward, next_state, done))
                    state = next_state
                    
                ### LEARNING PART            
                # Obtain random mini-batch from memory
                if firstp:
                    tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                else:
                    tree_idx, batch, ISWeights_mb = memory2.sample(batch_size)
                    
                states_mb = np.array([each[0][0] for each in batch])
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch]) 
                next_states_mb = np.array([each[0][3] for each in batch])
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []
                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                
                # Get Q values for next_state
                if firstp:
                    q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                else:
                    q_next_state = sess.run(DQNetwork2.output, feed_dict = {DQNetwork2.inputs_: next_states_mb})
                    
                # Calculate Qtarget for all actions that state
                if firstp:
                    q_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_: next_states_mb})
                else:
                    q_target_next_state = sess.run(TargetNetwork2.output, feed_dict = {TargetNetwork2.inputs_: next_states_mb})
                    
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a') 
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    
                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])


                if firstp:
                    _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                                   DQNetwork.targetQ: targets_mb,
                                                                   DQNetwork.actions_: actions_mb,
                                                                   DQNetwork.ISWeights_: ISWeights_mb})
                    # Update priority
                    memory.batch_update(tree_idx, absolute_errors)
                else:
                    _, loss, absolute_errors = sess.run([DQNetwork2.optimizer, DQNetwork2.loss, DQNetwork2.absolute_errors],
                                                        feed_dict={DQNetwork2.inputs_: states_mb,
                                                                   DQNetwork2.targetQ: targets_mb,
                                                                   DQNetwork2.actions_: actions_mb,
                                                                   DQNetwork2.ISWeights_: ISWeights_mb})
                    # Update priority
                    memory2.batch_update(tree_idx, absolute_errors)
                    
                
                
                if tau > max_tau and done:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    firstp = not firstp
                    print("Model updated")
                    break

            # Save model every 1000 episodes
            if episode % 1000 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

                
with tf.Session() as sess:
    while True:
        saver.restore(sess, "./models/model.ckpt")
        evaluate = tf.keras.models.load_model('models/eval.h5')
        score = 0

        for i in range(100000):
            game.new_episode()
            firstp = True
            if display:
                game.print_cards()
            state = game.get_state()
            while True:
                if firstp:
                    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1,) + state.shape)})
                else:
                    Qs = sess.run(DQNetwork2.output, feed_dict = {DQNetwork2.inputs_: state.reshape((1,) + state.shape)})
                    
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]
                firstp = not firstp

                r = game.make_action(action)
                if display:
                    game.print_action(action)
                done = game.is_episode_finished()

                if done:
                    if display:
                        print("Score: ", r)
                    score += r
                    break
                else:
                    state = game.get_state()

            if display:
                sys.stdout.write("Play one more game? [Y/n] ")
                choice = raw_input().lower()
                if choice == "n":
                    break
        if display:
            break
        print('Average Score: ', score/100000)
