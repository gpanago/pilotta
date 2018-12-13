Use
./dataset number file
to generate data.

To train Model 1 and Model 2, you need to unpack data.tar.bz2 in the data/ directory.

Execute ./pilotta1.py to train Model 1.

Execute ./pilotta2.py to train Model 2.

The Q-learning algorithm is in pilotta_q.py

By default, it uses the trained AI to display bidding scenarios.
Change display=False to print an average score for the current AI.
Change learning=True to start learning for the biggining (note: overrides the previous model).

Run ./plot.py to create the plot of the Q-learning average scores.
