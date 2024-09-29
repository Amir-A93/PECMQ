import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import json

def ReadLog(content,ts, cl,r):
    sessions = content
    counter_dict = np.zeros((len(cl),len(ts)),dtype = int)
    print(len(content))

    for i in range(len(ts)):
        for j in range(len(cl)):
            for k in range(r):
                # print(sessions[k + (j*r) + (i*len(cl)*r)]['session_status'])
                if(sessions[k + (j*r) + (i*len(cl)*r)]['id1_verified'] == 1):
                     if(sessions[k + (j*r) + (i*len(cl)*r)]['id2_verified'] == 1):
                        counter_dict [j][i] += 1

    return counter_dict


def plot3D(client_stats,ts,cl):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    topic_sizes = np.array(ts)
    code_length = np.array(cl)
    X, Y = np.meshgrid(code_length,topic_sizes)
    cs = client_stats/10.0
    surf = ax.plot_surface(X, Y, cs, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_zlim(0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xticks(np.arange(1,len(code_length)+1))
    ax.set_yticks(np.arange(16,208,16))
    ax.set_xlabel("repetition code length")
    ax.set_ylabel("topic size (No. bits)")
    ax.set_zlabel('success rate')
    plt.yticks(rotation=-45)
    plt.show()


# def plotCurve(client_stats,ts,cl):
#     plt.rcParams.update({'font.size': 14})
#     fig, ax = plt.subplots()
#     fig.set_figwidth(15)
#     fig.set_figheight(4)
#     for i in range(len(client_stats)):
#         #ax.plot(client_stats[i]/1000.0,label=str(ts[i]),marker="o")
#         ax.plot(client_stats[i]/10.0,label=str(cl[i]),marker="o")
#     ax.set_xticks(np.arange(0,len(ts)))
#     ax.set_xticklabels(ts)
#     ax.set_ylim(-0.05,1.1)
#     #ax.legend(title="topic size (bits)")
#     ax.legend(title="rep length (bits)", loc=(1.01,0))
#     ax.set_xlabel("nonce size (bits)")
#     ax.set_ylabel("success rate")
    
#     plt.axhline(y = 0.63, color = 'r', linestyle = '--') 
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     # ax.text(0.1, 0.7, "rep length > 2", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
#     # ax.text(0.8, 0.5, "rep length 1 & 2", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
#     plt.grid()
#     plt.savefig("reliability.png",bbox_inches='tight',dpi=100)
#     plt.savefig("reliability.pdf",bbox_inches='tight')
#     plt.show()
    
    

rep = 10
ts = topic_sizes = [16,32,64,96,128,192]
cl1 = code_length1 = ['(15,7)','(63,7)']
cl2 = code_length2 = ['(15,5)']

c_log_path1 = "ttp_sessions.json"
session_log_file1 = open(c_log_path1, 'r')
content1 = json.load(session_log_file1)
client_stats1 = ReadLog(content1,topic_sizes,code_length1,rep)
c_log_path2 = "code515/ttp_sessions.json"
session_log_file2 = open(c_log_path2, 'r')
content2 = json.load(session_log_file2)
client_stats2 = ReadLog(content2,topic_sizes,code_length2,rep)


# print(client_stats)
# plot3D(client1_stats,topic_sizes,code_length)
# plotCurve(client_stats,topic_sizes,code_length)

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(4)
for i in range(len(client_stats1)):
    #ax.plot(client_stats[i]/1000.0,label=str(ts[i]),marker="o")
    ax.plot(client_stats1[i]/10.0,label=str(cl1[i]),marker="o")
for i in range(len(client_stats2)):
    #ax.plot(client_stats[i]/1000.0,label=str(ts[i]),marker="o")
    ax.plot(client_stats2[i]/10.0,label=str(cl2[i]),marker="o")

ax.set_xticks(np.arange(0,len(ts)))
ax.set_xticklabels(ts)
ax.set_ylim(-0.05,1.1)
#ax.legend(title="topic size (bits)")
ax.legend(title="bch code", loc=(1.01,0))
ax.set_xlabel("nonce size (bits)")
ax.set_ylabel("success rate")

plt.axhline(y = 0.75, color = 'r', linestyle = '--') 
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.1, 0.7, "rep length > 2", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
# ax.text(0.8, 0.5, "rep length 1 & 2", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

plt.grid()
plt.savefig("reliability.png",bbox_inches='tight',dpi=100)
plt.savefig("reliability.pdf",bbox_inches='tight')
plt.show()
    