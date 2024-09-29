import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import json
    

rep = 10
topic_sizes = [16,32,64,96,128,192]
code_length = [3,5,9]

latencies = [[[],[],[]],
             [[],[],[]],
             [[],[],[]],
             [[],[],[]],
             [[],[],[]],
             [[],[],[]]]

for cl in range(len(code_length)):
    ttp_log_file = open("KH_ttp_sessions_rep" + str(code_length[cl]) + ".json",mode='r')
    stats = json.load(ttp_log_file)
    ttp_log_file.close()
    for ts in range(len(topic_sizes)):
        for i in range(len(stats)):
            if(stats[i]['session_status'] == 'established'):
                if(stats[i]['key_size'] == topic_sizes[ts]):
                    if(stats[i]['replen'] == code_length[cl]):
                        val = stats[i]['delta_time'] - stats[i]['puf_time']
                        latencies[ts][cl].append(val)

# print(latencies)


ticks = ['3', '5', '9']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    # plt.setp(bp['whiskers'], color=color)
    # plt.setp(bp['caps'], color=color)
    # plt.setp(bp['medians'], color='red')

plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 14})

b1 = plt.boxplot(latencies[0], positions=np.array(range(len(latencies[0])))*2.0-0.5, sym='',vert=True,  widths=0.15,patch_artist=True)
b2 = plt.boxplot(latencies[1], positions=np.array(range(len(latencies[1])))*2.0-0.3, sym='',vert=True,  widths=0.15,patch_artist=True)
b3 = plt.boxplot(latencies[2], positions=np.array(range(len(latencies[2])))*2.0-0.1, sym='',vert=True,  widths=0.15,patch_artist=True)
b4 = plt.boxplot(latencies[3], positions=np.array(range(len(latencies[3])))*2.0+0.1, sym='',vert=True,  widths=0.15,patch_artist=True)
b5 = plt.boxplot(latencies[4], positions=np.array(range(len(latencies[4])))*2.0+0.3, sym='',vert=True,  widths=0.15,patch_artist=True)
b6 = plt.boxplot(latencies[5], positions=np.array(range(len(latencies[5])))*2.0+0.5, sym='',vert=True,  widths=0.15,patch_artist=True)


# colors are from http://colorbrewer2.org/
set_box_color(b1, '#c6dbef') 
set_box_color(b2, '#9ecae1')
set_box_color(b3, '#6baed6') 
set_box_color(b4, '#4292c6')
set_box_color(b5, '#2171b5') 
set_box_color(b6, '#084594')

# draw temporary red and blue lines and use them to create a legend

plt.plot([], c='#c6dbef', label='nonce size 16 bits')
plt.plot([], c='#9ecae1', label='nonce size 32 bits')
plt.plot([], c='#6baed6', label='nonce size 64 bits')
plt.plot([], c='#4292c6', label='nonce size 96 bits')
plt.plot([], c='#2171b5', label='nonce size 128 bits')
plt.plot([], c='#084594', label='nonce size 192 bits')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
# plt.xlim(-2, len(ticks)*2)
plt.ylim(0, 250)
plt.xlabel("Repetition code length")
plt.ylabel("Key establishment latency (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig('pecmq_latency.pdf')
plt.show()