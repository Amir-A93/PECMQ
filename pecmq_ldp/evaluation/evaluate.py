import numpy as np
import json

original_set_file = open("./original/crpset.json","r")
ldpnoisy_set_file = open("./noisy/crpset.json","r")

original_set = json.load(original_set_file)
ldp_noisy_set = json.load(ldpnoisy_set_file)

set_size = len(original_set)

count = 0

for i in range(set_size):
    if(original_set[str(i)]["response"] == ldp_noisy_set[str(i)]["response"]):
        count += 1
print("Number of flipped responses: " + str(set_size - count) + "/" + str(set_size))


print("::::::::::::::::::::::::Evaluating a randomly selected subset::::::::::::::::::::::::")
count_one_original = 0
count_one_noisy = 0
count = 0
random_bound_size = 20
# random_bound_size = set_size
random_bound_s = np.random.randint(0,set_size - random_bound_size)
# random_bound_s = 0
print("Random bound size: " + str(random_bound_size))
print("Random bound starting index: " + str(random_bound_s))
for j in range(random_bound_s,random_bound_s + random_bound_size):
    if(original_set[str(j)]["response"] == ldp_noisy_set[str(j)]["response"]):
        count += 1

    if(original_set[str(j)]["response"] == 1.0):
        count_one_original += 1
    if(ldp_noisy_set[str(j)]["response"] == 1.0):
        count_one_noisy += 1
    
print("Number of flipped responses in random bound: " + str(random_bound_size - count) + "/" + str(random_bound_size))
print("Number of ones in original set: " + str(count_one_original) + "/" + str(random_bound_size))
print("Number of ones in noisy set: " + str(count_one_noisy) + "/" + str(random_bound_size))


for subset in range(0, set_size, 20)