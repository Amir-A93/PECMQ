import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import json
    

rep = 10
topic_sizes = [32,64,96,128,192]
code_length = [5,7,9]

# latencies = [[[],[],[]],
#              [[],[],[]],
#              [[],[],[]],
#              [[],[],[]],
#              [[],[],[]],
#              [[],[],[]]]

latencies1 = [[[],[],[],[],[],[]],
             [[],[],[],[],[],[]],
             [[],[],[],[],[],[]]]

memusage1 = [[0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0]]

for cl in range(len(code_length)):
    ttp_log_file = open("evals/kh_ttp_rp_n2_sessions.json",mode='r')
    stats = json.load(ttp_log_file)
    ttp_log_file.close()
    for ts in range(len(topic_sizes)):
        count = 0
        for i in range(len(stats)):
            if(stats[i]['session_status'] == 'established'):
                if(stats[i]['key_size'] == topic_sizes[ts]):
                    if(stats[i]['replen'] == code_length[cl]):
                        # val = stats[i]['delta_time'] - stats[i]['puf_time']
                        val = stats[i]['mempeak']
                        # latencies1[cl][ts].append(val)
                        memusage1[cl][ts] += val
                        count += 1
        memusage1[cl][ts] = memusage1[cl][ts] / count


code_length2 = [15,63]
latencies2 = [[[],[],[],[],[],[]],
             [[],[],[],[],[],[]]]

memusage2 = [[0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0]]

for cl in range(len(code_length2)):
    ttp_log_file = open("evals/kh_ttp_bch_7_n2_sessions.json",mode='r')
    stats = json.load(ttp_log_file)
    ttp_log_file.close()
    for ts in range(len(topic_sizes)):
        count = 0
        for i in range(len(stats)):
            if(stats[i]['session_status'] == 'established'):
                if(stats[i]['key_size'] == topic_sizes[ts]):
                    if(stats[i]['replen'] == code_length2[cl]):
                        # val = stats[i]['delta_time'] - stats[i]['puf_time']
                        val = stats[i]['mempeak']
                        # latencies2[cl][ts].append(val)
                        memusage2[cl][ts] += val
                        count += 1
        memusage1[cl][ts] = memusage2[cl][ts] / count

code_length3 = [15]
latencies3 = [[[],[],[],[],[],[]]]
memusage3 = [[0.0,0.0,0.0,0.0,0.0]]

for cl in range(len(code_length3)):
    ttp_log_file = open("evals/kh_ttp_bch_5_n2_sessions.json",mode='r')
    stats = json.load(ttp_log_file)
    ttp_log_file.close()
    for ts in range(len(topic_sizes)):
        count = 0
        for i in range(len(stats)):
            if(stats[i]['session_status'] == 'established'):
                if(stats[i]['key_size'] == topic_sizes[ts]):
                    if(stats[i]['replen'] == code_length3[cl]):
                        # val = stats[i]['delta_time'] - stats[i]['puf_time']
                        val = stats[i]['mempeak']
                        # latencies3[cl][ts].append(val)
                        memusage3[cl][ts] += val
                        count += 1
        memusage3[cl][ts] = memusage3[cl][ts] / count


ticks = ['32', '64', '96', '128', '192']
x = np.arange(len(ticks))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

# def set_box_color(bp, color):
#     plt.setp(bp['boxes'], color=color)
#     # plt.setp(bp['whiskers'], color=color)
#     # plt.setp(bp['caps'], color=color)
#     # plt.setp(bp['medians'], color='red')

plt.figure(figsize=(12,5))
plt.rcParams.update({'font.size': 14})

# b1 = plt.boxplot(latencies1[0], positions=np.array(range(6))*2.0-0.5, sym='',vert=True,  widths=0.15,patch_artist=True)
# b2 = plt.boxplot(latencies1[1], positions=np.array(range(6))*2.0-0.3, sym='',vert=True,  widths=0.15,patch_artist=True)
# b3 = plt.boxplot(latencies1[2], positions=np.array(range(6))*2.0-0.1, sym='',vert=True,  widths=0.15,patch_artist=True)
# b4 = plt.boxplot(latencies2[0], positions=np.array(range(6))*2.0+0.1, sym='',vert=True,  widths=0.15,patch_artist=True)
# b5 = plt.boxplot(latencies3[0], positions=np.array(range(6))*2.0+0.3, sym='',vert=True,  widths=0.15,patch_artist=True)
# b6 = plt.boxplot(latencies2[1], positions=np.array(range(6))*2.0+0.5, sym='',vert=True,  widths=0.15,patch_artist=True)
# plt.bar()
offset = width * -.5
b1 = plt.bar(x + offset,memusage1[0],width,color='#4292c6')
# plt.bar_label(b1, padding=3)
offset = width * 0.7
b2 = plt.bar(x + offset,memusage1[1],width,color='#2171b5')
offset = width * 1.9
b3 = plt.bar(x + offset,memusage1[2],width,color='#084594')
offset = width * 3.1
b4 = plt.bar(x + offset,memusage2[0],width,color='#fee6ce')
offset = width * 4.3
b5 = plt.bar(x + offset,memusage3[0],width,color='#fdae6b')
offset = width * 5.5
b6 = plt.bar(x + offset,memusage2[1],width,color='#e6550d')

print(memusage3)
# print(latencies2[1])
# colors are from http://colorbrewer2.org/

# set_box_color(b1, '#4292c6')
# set_box_color(b2, '#2171b5') 
# set_box_color(b3, '#084594')
# set_box_color(b4, '#fee6ce') 
# set_box_color(b5, '#fdae6b')
# set_box_color(b6, '#e6550d') 

# draw temporary red and blue lines and use them to create a legend

plt.plot([], c='#4292c6', label='rep_rep-len=5')
plt.plot([], c='#2171b5', label='rep_rep-len=7')
plt.plot([], c='#084594', label='rep_rep-len=9')
plt.plot([], c='#fee6ce', label='bch_code=(15,7)')
plt.plot([], c='#fdae6b', label='bch_code=(15,5)')
plt.plot([], c='#e6550d', label='bch_code=(63,7)')
plt.legend()

# plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xticks(x + (width*2), ticks)
# plt.xlim(-2, len(ticks)*2)
plt.ylim(0, 80)
plt.xlabel("nonce size (bits)")
plt.ylabel("Peak memory usage (mega-bytes)")
plt.grid(True)
plt.tight_layout()
plt.savefig('pecmq_mempeak.pdf')
plt.show()