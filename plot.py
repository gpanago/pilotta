#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

a = []
for i in range(4):
    with open('results.txt' + str(i), 'r') as f:
        b = []
        for l in f:
            if 'Average Score' in l:
                l = l.split(',')[1].split(')')[0]
                b.append(float(l))
        a.append(b)


l = min([len(x) for x in a])
fig, ax = plt.subplots()
plt.plot([0,l],[0,0],'--y')
for i in range(4):
    plt.plot(range(l), a[i][:l])
ax.set_xlabel('Time', fontsize=18)
plt.xticks(fontsize=14)

ax.set_ylabel('Average score', fontsize=18)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('fig.pdf')
print(l)


