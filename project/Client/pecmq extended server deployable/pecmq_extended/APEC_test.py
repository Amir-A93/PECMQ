import numpy as np
from ruhrmair_python.PUFmodels import XORArbPUF, linArbPUF
import pickle
import os
from os import path



def Create_PUF(num_bits,num_xor):
        puf_dir =  "APEC_PUF"
        if(path.exists(puf_dir) != True):
            os.mkdir(puf_dir)   
        puf_instance = XORArbPUF(num_bits=num_bits,
                                      numXOR=num_xor,
                                      type='equal')
        
        with open(puf_dir + "/puf_model.pkl", 'wb') as output:
            pickle.dump(puf_instance, output, pickle.HIGHEST_PROTOCOL) 
        return puf_instance


def concat_challenge(challenge_size, xor_size, challenges):
        new_challenges = np.empty([xor_size, challenge_size, challenges.shape[1]])
        for puf in range(xor_size):
            new_challenges[puf] = challenges
        return new_challenges

def generate_challenge(numCRPs,challenge_size):
        challenges = np.random.randint(0, 2, [challenge_size, numCRPs])      
        return challenges


challenge_size = 64
xor_size = 2
nonce = [1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0] #Consider rep coidng with length 2, thus 8 tokens required
num_required_direct_tokens = len(nonce)/2
num_required_inverse_tokens = len(nonce)/2

puf_instance = Create_PUF(challenge_size,xor_size)

direct_count = 0
inverse_count = 0
elected_challenges = []
indeces = []
polarity = []

for i in range(1000):

    challenge = generate_challenge(1,challenge_size)
    new_challenge = concat_challenge(challenge_size,xor_size,challenge)
    feature = puf_instance.calc_features(new_challenge)
    response = puf_instance.bin_response(feature)

    copy_challenge = challenge
    for i in range(challenge_size):
        
        ch_bit = copy_challenge[i]
        # copy_challenge[i] = 1.0 - copy_challenge[i]
        if(copy_challenge[i] == 1.0):
              copy_challenge[i] = 0.0
        elif(copy_challenge[i] == 0.0):
              copy_challenge[i] = 1.0

        new_challenge_copy = concat_challenge(challenge_size,xor_size,copy_challenge)
        feature_copy = puf_instance.calc_features(new_challenge_copy)
        response_copy = puf_instance.bin_response(feature_copy)

        if(response == response_copy):
              continue
        else:
            if(((ch_bit == 0) and (response == 1.)) or ((ch_bit == 1) and (response == -1.))):
                # print("Direct token found")
                if(direct_count< num_required_direct_tokens):
                        if(elected_challenges == []):
                              elected_challenges = np.array([copy_challenge])
                        else:
                              elected_challenges = np.append(elected_challenges,[copy_challenge],axis=0)
                        indeces.append(i)
                        polarity.append(1)
                        direct_count +=1
                else:
                        print("Direct count is full")
            elif(((ch_bit == 0) and (response == -1.)) or ((ch_bit == 1) and (response == 1.))):
                # print("Inverse token found")
                if(inverse_count< num_required_inverse_tokens):
                        if(elected_challenges == []):
                              elected_challenges = np.array([copy_challenge])
                        else:
                              elected_challenges = np.append(elected_challenges,[copy_challenge],axis=0)
                        indeces.append(i)
                        polarity.append(-1)
                        inverse_count +=1
                else:
                        print("Inverse count is full")
            if((inverse_count == num_required_inverse_tokens) and (direct_count == num_required_direct_tokens)):
                print("Found all tokens!")
                break
    if((inverse_count == num_required_inverse_tokens) and (direct_count == num_required_direct_tokens)):
        print("Found all tokens!")
        break


# print(elected_challenges.shape)
# print(indeces)
print(polarity)

elected_challenges = np.squeeze(elected_challenges)
for i,x in enumerate(nonce):
      elected_challenges[i][indeces[i]] = float(x)

elected_challenges = elected_challenges.transpose()
concated_challenges = concat_challenge(challenge_size,xor_size,elected_challenges)

# print(elected_challenges.shape)
features = puf_instance.calc_features(concated_challenges)
resps = puf_instance.bin_response(features)
# print(resps)

decoded_resps = np.zeros(len(resps),dtype=int)
for i in range(len(resps)):
    if(polarity[i] == -1):
        if(resps[i] < 0):
                decoded_resps[i] = 1
        else:
                decoded_resps[i] = 0
    else:
        if(resps[i] < 0):
                decoded_resps[i] = 0
        else:
                decoded_resps[i] = 1

print("Original nonce:")
print(nonce)
print("Decoded nonce:")
print(decoded_resps)
# dummy = generate_challenge(10,challenge_size)
# print(dummy.shape)