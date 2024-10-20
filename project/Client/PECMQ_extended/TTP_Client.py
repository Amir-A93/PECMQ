from executable_class import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import os
from os import path
import pickle
import pandas as pd
import sys 
from io import StringIO
import json
import hashlib
import string
import random
import galois
from reedsolo import RSCodec

class TTP_Client(PubSub_Base_Executable): ##CHANGE:: change class name
    
    ##Here you put the class specific variables
    ##...
    ##_________________________________________

    def __init__(self,
                 myID,
                 broker_ip,
                 broker_port,
                 introduction_topic,
                 controller_executable_topic,
                 controller_echo_topic, 
                 start_loop): 
                ##CHANGE:: --> , <<Extended Parameters>> ...):
        
        
        ## Here you initialize the class specific variables
        self.sessions = [] # [owner_id, partner_id, topic]
        self.topic_size = 32
        self.code = None
        ##_________________________________________________

        ##IMPORTANT:: This line of code is needed to append the name of newly defined class specific executables::
        self.executables.append('enroll_device')
        self.executables.append('enroll_device_ldp')
        self.executables.append('establish_private_topic')
        self.executables.append('verify_hash')

       
        ## ____________________________________________________________________________________________

        #IMPORTANT:: DON'T CHANGE:: Let the base class initializer be at the buttom. This is for the case if client start loop is set to start rightaway.
        PubSub_Base_Executable.__init__(self,
                                        myID,
                                        broker_ip,
                                        broker_port,
                                        introduction_topic,
                                        controller_executable_topic,
                                        controller_echo_topic,
                                        start_loop) ## DON'T CHANGE
        
        self.client_to_TTP_topic = "C2TTP"
        self.TTP_to_client_topic = "TTP2C"
        self.client.subscribe(self.client_to_TTP_topic)
        ##____________________________________________________________________________________________________________________________________________________

    def execute_on_msg (self,client,userdata, msg):
        PubSub_Base_Executable.execute_on_msg(self,client,userdata,msg)

        # try:

        header_body = str(msg.payload.decode()).split('::')
        header_parts = header_body[0].split('|')
        
        ##IMPORTANT:: Here you extend the message parser to check for the class specific executables
        ## CHANGE:: --> if msg_parts[0] == <<executable function name>>:
            ##Here you execute accordingly, or simply just invoke the executable: example: self.<<executable name>>(parameterst)
        if header_parts[2] == 'enroll_device':
            id = header_body[1].split('-id ')[1].split(' -csize ')[0]
            csize = int(header_body[1].split('-csize ')[1].split(' -nxor ')[0])
            nxor = int(header_body[1].split('-nxor ')[1].split(' -crp ')[0])
            crp = header_body[1].split(' -crp ')[1].split(';')[0]
            self.enroll_device(id,crp,csize,nxor)
        
        if header_parts[2] == 'enroll_device_ldp':
            id = header_body[1].split('-id ')[1].split(' -csize ')[0]
            csize = int(header_body[1].split('-csize ')[1].split(' -nxor ')[0])
            nxor = int(header_body[1].split('-nxor ')[1].split(' -crp ')[0])
            crp = header_body[1].split(' -crp ')[1].split(';')[0]
            self.enroll_device_ldp(id,crp,csize,nxor)

        elif header_parts[2] == 'establish_private_topic':
            id1 = header_body[1].split('-id1 ')[1].split(' -id2 ')[0]                
            id2 = header_body[1].split(' -id2 ')[1].split(' -l ')[0]
            self.topic_size = int(header_body[1].split(' -l ')[1].split(' -rl ')[0])
            replen          = int(header_body[1].split(' -rl ')[1].split(';')[0])
            self.establish_private_topic(id1,id2,replen)
        
        elif header_parts[2] == 'verify_hash':
            id = header_body[1].split('-id ')[1].split(' -hash ')[0]     
            hash = header_body[1].split('-hash ')[1].split(' -tpuf ')[0]                  
            tpuf = header_body[1].split('-tpuf ')[1].split(' -pubtime ')[0]       
            transmissionT = str(header_body[1].split(' -pubtime ')[1].split(' -memavg ')[0])
            memavg = str(header_body[1].split(' -memavg ')[1].split(' -mempeak ')[0])
            mempeak = str(header_body[1].split(' -mempeak ')[1].split(' -cput ')[0])
            cput = str(header_body[1].split(' -cput ')[1].split(';')[0])
            self.verify_hash(id,hash,tpuf,memavg, mempeak, cput)

            ##__________________________________________________________________________________________________________________
            ##________________________________________________________________________________
        # except Exception as e:
        #     print("Error occured or message was not right!")
        #     print("ERROR: " + str(e))
            
    ##Here you define the new class specific executables:
    def enroll_device(self, id, crp_str,csize,nxor):
        print("Enrolling " + str(id))
        crp_str_parts = crp_str.split('&')
        meta_shape = crp_str_parts[0].split(' ')
        challenge_str = crp_str_parts[1]
        resp_str = crp_str_parts[2]
        
        challenge_set = np.fromstring(challenge_str,dtype=int,sep=' ')
        challenge_set = challenge_set.reshape(int(meta_shape[1]),int(meta_shape[2]))
        
        response_set = np.fromstring(resp_str,dtype=int,sep=' ')
        print(challenge_set.shape)
        print(response_set.shape)
        acc = self.train_model(set_size=challenge_set.shape[0],
                         challenge_size=challenge_set.shape[1]-1,
                         xor_size=nxor,
                         challenge_set=challenge_set,
                         responses=response_set,
                         client_id=id)
        
        client_dir =  "CLIENT_" + id
        if(path.exists(client_dir) != True):
            os.mkdir(client_dir)

        client_info = {
            "id": id,
            "csize" : csize,
            "nxor" : nxor,
            "model_acc": acc
        }

        with open(client_dir + "/client_info.json", 'w') as output:
            json.dump(client_info,output)

        self.echo_msg("Client " + id + " has been enrolled with model accuracy = " + str(acc))

    def enroll_device_ldp(self, id, crp_str,csize,nxor):
        print("Enrolling " + str(id))
       
        client_dir =  "CLIENT_" + id
        if(path.exists(client_dir) != True):
            os.mkdir(client_dir)

        client_info = {
            "id": id,
            "csize" : csize,
            "nxor" : nxor
        }

        with open(client_dir + "/client_info.json", 'w') as output:
            json.dump(client_info,output)

        with open(client_dir + "/client_ldp_dataset.json", 'w') as output:
            json.dump(crp_str,output)

        self.echo_msg("Client " + id + " has been enrolled with mlv")


    def verify_hash(self,id, hash,puf_time,memavg,mempeak,cput):
        for i in range(len(self.sessions)):

            if (id == self.sessions[i]['id1']):
                if(hash == self.sessions[i]['topic_hash']):
                    if(self.sessions[i]['id1_verified'] == 1):
                        continue
                    else:
                        self.sessions[i]['id1_verified'] = 1
                        if(self.sessions[i]['id2_verified'] == 1):
                            self.sessions[i]['session_status']='established'
                            presentDate = datetime.datetime.now()
                            unix_timestamp = datetime.datetime.timestamp(presentDate)*1000
                            self.sessions[i]['time_of_establishment'] = unix_timestamp
                            self.sessions[i]['delta_time'] = unix_timestamp - float(self.sessions[i]['time_of_request'])
                            self.sessions[i]['puf_time'] = float(self.sessions[i]['puf_time']) + float(puf_time)

                            self.sessions[i]['memavg'] = (float(self.sessions[i]['memavg']) + float(memavg))/2
                            self.sessions[i]['mempeak'] = (float(self.sessions[i]['mempeak']) + float(mempeak))/2
                            self.sessions[i]['cput'] = (float(self.sessions[i]['cput']) + float(cput))/2
                        else:
                            self.sessions[i]['puf_time'] = float(puf_time)
                            self.sessions[i]['memavg'] =  float(memavg)
                            self.sessions[i]['mempeak'] =  float(mempeak)
                            self.sessions[i]['cput'] =  float(cput)
                        break
                        # self.publish("TTP2"+str(id),"ack_hash","-res 1 -h " + str(hash))
                else:
                    if(self.sessions[i]['id1_verified'] != 0):
                        continue
                    else:
                        self.sessions[i]['id1_verified'] = -1
                        self.sessions[i]['session_status']='failed'
                        presentDate = datetime.datetime.now()
                        unix_timestamp = datetime.datetime.timestamp(presentDate)*1000
                        self.sessions[i]['time_of_establishment'] = unix_timestamp
                        self.sessions[i]['delta_time'] = unix_timestamp - float(self.sessions[i]['time_of_request'])
                        # self.publish("TTP2"+str(id),"ack_hash","-res -1 -h " + str(hash))
                      
                    
            elif (id == self.sessions[i]['id2']):
                if(hash == self.sessions[i]['topic_hash']):
                    if(self.sessions[i]['id2_verified'] == 1):
                        continue
                    else:
                        self.sessions[i]['id2_verified'] = 1
                        if(self.sessions[i]['id1_verified'] == 1):
                            self.sessions[i]['session_status']='established'
                            presentDate = datetime.datetime.now()
                            unix_timestamp = datetime.datetime.timestamp(presentDate)*1000
                            self.sessions[i]['time_of_establishment'] = unix_timestamp
                            self.sessions[i]['delta_time'] = unix_timestamp - float(self.sessions[i]['time_of_request'])
                            self.sessions[i]['puf_time'] = float(self.sessions[i]['puf_time']) + float(puf_time)
                            self.sessions[i]['memavg'] = (float(self.sessions[i]['memavg']) + float(memavg))/2
                            self.sessions[i]['mempeak'] = (float(self.sessions[i]['mempeak']) + float(mempeak))/2
                            self.sessions[i]['cput'] = (float(self.sessions[i]['cput']) + float(cput))/2
                        else:
                            self.sessions[i]['puf_time'] = float(puf_time)
                            self.sessions[i]['memavg'] =  float(memavg)
                            self.sessions[i]['mempeak'] =  float(mempeak)
                            self.sessions[i]['cput'] =  float(cput)
                        break
                        # self.publish("TTP2"+str(id),"ack_hash","-res 1 -h " + str(hash))
                else:
                    if(self.sessions[i]['id2_verified'] != 0):
                        continue
                    else:
                        self.sessions[i]['id2_verified'] = -1
                        self.sessions[i]['session_status']='failed'
                        presentDate = datetime.datetime.now()
                        unix_timestamp = datetime.datetime.timestamp(presentDate)*1000
                        self.sessions[i]['time_of_establishment'] = unix_timestamp
                        self.sessions[i]['delta_time'] = unix_timestamp - float(self.sessions[i]['time_of_request'])
                        # self.publish("TTP2"+str(id),"ack_hash","-res -1 -h " + str(hash))
                        break
        
        with open(self.id + '_sessions.json', 'w') as json_file:  
           json.dump(self.sessions, json_file)

    def get_random_string(self,length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def establish_private_topic(self, id1, id2,replen):
        print("Establishing private topic for whispering between " + str(id1) +" and " + str(id2))
        
        client1_info = self.load_client(id1)
        client2_info = self.load_client(id2)



        key_bin = np.random.randint(0,2,self.topic_size,dtype=int) 
        key_bin = self.appendzeros_bch(key_bin)
        topic_str = ""
        for i in key_bin:
            topic_str += str(int(i))
        
        # key = self.get_random_string(self.topic_size/8)
              
        # key_bin = ''
        # for ch in key:
        #     x = bin(ord(ch)).split('b')[1]
        #     if(len(x) == 7):
        #         x = '0'+x
        #     key_bin += x
        
        topic_hash = hashlib.sha256(topic_str.encode()).hexdigest()
        
        # print(key)
        # print(key_bin)
     
        presentDate = datetime.datetime.now()
        unix_timestamp = datetime.datetime.timestamp(presentDate)*1000

        #self.sessions.append([id1,id2,topic_str,topic_hash])
        self.sessions.append({'id1':id1,
                              'id2':id2,
                              'time_of_request': unix_timestamp,
                              'key':topic_str,
                              'key_size':self.topic_size,
                              'replen':replen,
                              'topic_hash':str(topic_hash),
                              'id1_verified':0,
                              'id2_verified':0,
                              'session_status':'pending',
                              'time_of_establishment':-1,
                              'delta_time':-1,
                              'puf_time': -1})
        #print(self.sessions)
        with open(self.id + '_sessions.json', 'w') as json_file:  
           json.dump(self.sessions, json_file)

        # id1_CSet = self.match_making(topic_draft=topic_str,id=id1,rl=replen,cs = client1_info['csize'])
        # id2_CSet = self.match_making(topic_draft=topic_str,id=id2,rl=replen,cs = client2_info['csize'])
        
        id1_CSet = self.match_making(topic_draft=key_bin,id=id1,rl=replen,cs = client1_info['csize'])
        id2_CSet = self.match_making(topic_draft=key_bin,id=id2,rl=replen,cs = client2_info['csize'])

        cset1_msg = json.dumps(id1_CSet)
        cset2_msg = json.dumps(id2_CSet)

        cset1_msg_encryped = self.encrypt_msg(cset1_msg.encode())
        cset2_msg_encryped = self.encrypt_msg(cset2_msg.encode())
        
        # print(cset1_msg_encryped.decode('utf-8'))
        
        self.publish("TTP2"+str(id1),"register_private_topic",' -id ' + str(id1) + ' -c ' + str(id2) + ' -rl ' + str(replen) + ' -cset ' + cset1_msg_encryped.decode('utf-8'))
        self.publish("TTP2"+str(id2),"subscribe_to_private_topic",' -id ' + str(id2) +  ' -rl ' + str(replen) + ' -cset ' + cset2_msg_encryped.decode('utf-8'))
        
        #self.publish(self.TTP_to_client_topic,"subscribe_to_private_topic",' -id ' + str(id2) + ' -topic ' + wisper_topic)
        
    ##___________________________________________________

    def load_client(self,id):
        client_dir =  "CLIENT_" + id
        if(path.exists(client_dir) != True):
            print("ERROR: Client directory not found!")

        with open(client_dir + "/client_info.json", 'r') as output:
            client_info = json.load(output)
        return client_info
    
    def generate_challenge(self, numCRPs,challenge_size):
        challenges = np.random.randint(0, 2, [challenge_size, numCRPs])      
        return challenges
    
    def calc_features(self, challenges,challenge_size):
        # calculate feature vector of linear model
        temp = [np.prod(1 - 2 * challenges[i:, :], 0) for i in range(challenge_size)]
        features = np.concatenate((temp, np.ones((1, challenges.shape[1]))))
        return features

    def infer_with_model(self,id,challenge_size,set_size):
        challenges = self.generate_challenge(set_size,challenge_size)
        feature_set = self.calc_features(challenges,challenge_size)
        feature_set    = np.transpose(feature_set,(1,0))
        feature_tensor = torch.tensor(feature_set,dtype=torch.float32)
        model = self.load_nn_model(id)  
        responses = torch.floor(model(feature_tensor) + 0.5)
        challenge_transposed = np.transpose(challenges,(1,0))
        # print(challenge_transposed.shape)
        return [challenge_transposed.tolist(), responses.tolist()]
    
    
    def binary_to_bytes(self,binary_array):
        """Convert a binary array to a list of bytes."""
        byte_array = bytearray()
        for i in range(0, len(binary_array), 8):
            byte = binary_array[i:i+8]
            byte_value = int(''.join(map(str, byte)), 2)
            # byte_value = int.to_bytes(byte_value,8,sys.byteorder)
            byte_array.append(byte_value)
        return bytes(byte_array)

    def bytes_to_binary(self,byte_array):
        """Convert a list of bytes back to a binary array."""
        binary_array = []
        for byte in byte_array:
            binary_array.extend([int(bit) for bit in format(byte, '08b')])
        return binary_array

    def encode_with_reed_solomon(self,binary_array, n_parity_symbols):
    
        # if(self.code == None):
        #     self.code = RSCodec(n_parity_symbols)
        # rsc = self.code

        rsc = RSCodec(n_parity_symbols)

        byte_data = self.binary_to_bytes(binary_array)
        encoded_data = rsc.encode(byte_data)
        encoded_binary_array = self.bytes_to_binary(encoded_data)
        return encoded_binary_array

    def encode_with_bch(self,binary_array,block_length):
        n = block_length  # Codeword length
        k = 5  # Message length
        # print("n: " + str(n))
        # if(self.code == None):
        #     self.code = galois.BCH(n, k)
        # bch = self.code
        bch = galois.BCH(n, k)
        msg = self.appendzeros_bch(binary_array)
        msg =  msg.reshape((int(len(msg)/k),k))
        msg = np.array(msg,dtype=np.uint8)
        encoded_data = bch.encode(msg)

        return encoded_data.reshape(encoded_data.shape[0]*encoded_data.shape[1])

    def appendzeros_bch(self,bin_array):
        k = 5
        msg = bin_array
        mod = len(msg)%k
        if(mod != 0):
            msg = np.append(msg,np.zeros(k - mod))
        return msg

    def repetition_encode(self, binary_array, n_repeats):    
        return np.repeat(binary_array, n_repeats)

 

    def match_making(self, topic_draft, id,cs,rl):
        
        challenge_size = cs
        encoding_param = rl

        encoded_topic = self.encode_with_bch(topic_draft,encoding_param)
        # encoded_topic = self.encode_with_bch(topic_draft,encoding_param)
        # encoded_topic = self.flip_bits(np.array(encoded_topic),2) #adds artificial noise to the encoded topic

        # print(encoded_topic)
        batch_size = len(encoded_topic) * 5
        challenge_pack = []
        [challenges, responses] = self.infer_with_model(id,challenge_size,batch_size)

        batch_empty = False
        taken_elements = []
        for topic_index in encoded_topic:
            for j in range(len(challenges)):
                try:
                    if(int(topic_index) == int(responses[j][0])):
                        challenge_pack.append(challenges.pop(j))
                        responses.pop(j)
                        taken_elements.append(j)
                        break
                except:
                    print("length of batch: " + str(len(responses)))
                    # break
        
        return challenge_pack

    def flip_bits(self, binary_array, percentage):
        
        num_bits = len(binary_array)
        num_flips = int(np.ceil(num_bits * (percentage / 100.0)))

        # Select random indices to flip
        flip_indices = np.random.choice(num_bits, num_flips, replace=False)

        # Flip the bits at the selected indices
        noisy_array = binary_array.copy()
        noisy_array[flip_indices] = 1 - noisy_array[flip_indices]

        return noisy_array


    
    # def match_making(self, topic_draft, id,cs,rl):
        
    #     challenge_size = cs
    #     repetition_length = rl
    #     batch_size = cs * rl * 5
    #     challenge_pack = []
    #     [challenges, responses] = self.infer_with_model(id,challenge_size,batch_size)

    #     batch_empty = False
    #     taken_elements = []
    #     for topic_index in topic_draft:
    #         for i in range(repetition_length):
    #             for j in range(len(challenges)):
    #                 try:
    #                     if(int(topic_index) == int(responses[j][0])):
    #                         challenge_pack.append(challenges.pop(j))
    #                         responses.pop(j)
    #                         taken_elements.append(j)
    #                         break
    #                 except:
    #                     print("j is: " + str(j))
    #                     print("length of batch: " + str(len(responses)))
    #                     # break
        
    #     return challenge_pack

    def generate_wisper_topic(self,id1,id2):
        
        return str(id1) + str(id2) # PLACEHOLDER!!!!
    
    ##Here you define the rest of the class logic:
    
    def load_nn_model(self,id):
        
        prob_model_instance_dir =  "CLIENT_" + id
        if(path.exists(prob_model_instance_dir) != True):
            os.mkdir(prob_model_instance_dir)
        
        with open(prob_model_instance_dir + "/nn_prob_model.pkl", 'rb') as model_file:
            nn_model = pickle.load(model_file)
            
        return nn_model


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            
    def train_model(self,set_size,challenge_size,xor_size,challenge_set,responses, client_id):

        batch_size = 200
        validation_set_size = int(set_size * 0.1)
        test_set_size = int(set_size * 0.2)

        max_iter = 3
        MaxNsteps = 200

        learning_rate = 0.01
        treshold_loss = 0.01
        accuracy_threshold = 0.98

        training_set_size = int(set_size * 0.7)
        # =============================================================================
        # PREPARING THE TRAINING AND TEST SETS:
        # =============================================================================
        x_train = np.array(challenge_set[0:training_set_size])
        x_train = torch.tensor(x_train,dtype=torch.float32)
        # x_train = torch.reshape(x_train,(1,x_train.shape[0],x_train.shape[1]))
        
        y_train = np.array(responses[0:training_set_size])
        y_train = torch.tensor(y_train,dtype=torch.float32)
        y_train = np.reshape(y_train,(y_train.shape[0],))
        
        x_val = np.array(challenge_set[training_set_size : training_set_size + validation_set_size ])

        x_val = torch.tensor(x_val,dtype=torch.float32)
        
        y_val = np.array(responses[training_set_size : training_set_size + validation_set_size ])
        y_val = torch.tensor(y_val,dtype=torch.float32)
        
        
        x_test = np.array(challenge_set[training_set_size + validation_set_size : training_set_size + validation_set_size + test_set_size ])
        x_test = torch.tensor(x_test,dtype=torch.float32)
        
        y_test = np.array(responses[training_set_size + validation_set_size : training_set_size + validation_set_size + test_set_size ])
        y_test = torch.tensor(y_test,dtype=torch.float32)
        y_test = np.reshape(y_test,(y_test.shape[0],))
    
        # =============================================================================
        # BUILDING THE NEURAL NETWORK
        # =============================================================================
        layers = [torch.nn.Linear(challenge_size + 1 , pow(2,(xor_size-1)),bias = True),
                    torch.nn.Tanh(),
                    torch.nn.Linear(pow(2,(xor_size-1)),pow(2,(xor_size)),bias = True),
                    torch.nn.Tanh(),
                    torch.nn.Linear(pow(2,(xor_size)),pow(2,(xor_size-1)),bias = True),
                    torch.nn.Tanh(),
                    torch.nn.Linear(pow(2,(xor_size-1)),1,bias = True),
                    torch.nn.Sigmoid()]
    
        new_model = torch.nn.Sequential(*layers)
        nn_model = new_model

       

        #prob_model_parameters = nn_model.parameters()
        # prob_model_size = 0
        # for param in nn_model.parameters(): #Calculating the size of the model in terms of number of weight values multiplied by 4 (as float32)
        #     if(len(param.shape) > 1):
        #         prob_model_size += (param.shape[0] * param.shape[1] * 4)
        
        losses = np.array([0],dtype=float)
        val_accs = np.array([0],dtype=float)
        loss_func = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(),lr=learning_rate)
        attempt = 0
        # =============================================================================
        # TRAINING THE MODEL:
        # =============================================================================
        while(attempt < max_iter):

            
            nn_model.apply(self.weights_init)

            
            loss = 100
            epoch = 0
            train_begin = time.time()
            print("Training Begins...")
            # for epoch in range(Nsteps):

            while (loss > treshold_loss):
                epoch_complete = False
                batch_start = 0
                while(epoch_complete == False):
                    
                    batch_step = batch_size
                    
                    if(batch_start + batch_size >= training_set_size):
                        epoch_complete = True
                        batch_step = training_set_size - batch_start
                    else:
                        batch_start += batch_size
                    # print(batch_start)    
                    x_train_slice = x_train[batch_start:batch_start+batch_step]
                    y_train_slice = y_train[batch_start:batch_start+batch_step]
                    output = nn_model(x_train_slice)

                    #print("Y_train shape is: " + str(y_train_slice.shape))
                    #print("output shape is: " + str(output.shape))
                    
                    output = output.flatten()
                    loss = loss_func(output,y_train_slice)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
        
                if(epoch%2 == 0):
                    step = int(epoch/2)
                    
                    val_outputs = nn_model(x_val)
                    val_outputs = torch.floor(val_outputs + 0.5)
                    val_acc = 0
                    for nt in range(validation_set_size):
                        if(val_outputs[nt] == y_val[nt]):
                            val_acc += 1    
                    val_acc = val_acc / validation_set_size    
                    
                    val_accs[step] = val_acc
                    
                    
                    losses[step] = loss.item()
                    
                    
                    print("TL is = %r , Step = %d => loss = %.3f & validation acc = %.3f" % (False, epoch, loss, val_acc))
                    
                epoch += 1
                    
                if (epoch >= MaxNsteps):
                    print("Reached maximum epoch at %d. Model did not reach loss value %f"%(MaxNsteps,treshold_loss))
                    break
                else:
                    if(epoch%2 == 0):
                        val_accs = np.append(val_accs,[0])
                        losses = np.append(losses,[0])
                    # if(loss > treshold_loss):
                    #     losses = np.append(losses,[0])
        
            val_accs = val_accs[0:len(val_accs)-1]
            losses = losses[0:len(losses) -1]
            train_end = time.time()
            tot = int(train_end - train_begin)
            print("Time of training in seconds: %d" %(tot))
            
            # model_parameters_trained_list = list(nn_model.parameters())
            # for  parami in range(len(model_parameters_trained_list)):
            #     model_parameters_trained_list[parami] = model_parameters_trained_list[parami].data.numpy()
            
            
            # losses = losses[0:epoch]
            # =============================================================================
            # COMPUTING THE TEST ACCURACY:
            # =============================================================================
            # num_tests = int((1-training_set_proportion) * set_size)
            num_tests = test_set_size
            # nn_model = nn_model.to('cpu')
            test_outputs = nn_model(x_test)
            test_outputs = torch.floor(test_outputs + 0.5)
            acc = 0
            for nt in range(num_tests):
                if(test_outputs[nt] == y_test[nt]):
                    acc += 1    
            acc = acc / num_tests    
            print("Test accuarcy is: %.3f in %d tests"%(acc, num_tests))
            
            # =============================================================================
            # MEASURING TEST ACCURACY WHETHER IT FITS WITH THE EXPECTATIONS OR NOT
            # =============================================================================
            if(acc >= accuracy_threshold):
                print("Achieved accuracy is good. Saving the model data.")
                break
            else:
                if(attempt >= max_iter):               
                    print("Maximum case iteration reached. Saving the latest model.")
                else:
                    attempt +=1
                    print("Achieved accuracy is not good. Saving the latest model.")

        # =============================================================================
        # SAVING MODEL DATA
        # =============================================================================
        
        prob_model_instance_dir =  "CLIENT_" + client_id
        
        if(path.exists(prob_model_instance_dir) != True):
            os.mkdir(prob_model_instance_dir)
        
        with open(prob_model_instance_dir + "/nn_prob_model.pkl", 'wb') as output:
            pickle.dump(nn_model, output, pickle.HIGHEST_PROTOCOL)

        prob_model_instance_dir = prob_model_instance_dir + "/dataset_" + str(set_size)
        if(path.exists(prob_model_instance_dir) != True):
            os.makedirs(prob_model_instance_dir)
            
        prob_model_instance_dir = prob_model_instance_dir + "/training_Set_size" + str(training_set_size)
        if(path.exists(prob_model_instance_dir) != True):
            os.makedirs(prob_model_instance_dir)
            
        #Saving the information :
        pd.DataFrame(losses).to_csv(prob_model_instance_dir + "/losses.csv")
        pd.DataFrame(val_accs).to_csv(prob_model_instance_dir + "/val_accs.csv")
        
        stats = "Time of Training in seconds: " + str(tot) + "\n" +\
                "Train CRP size: " + str(training_set_size) + "\n" +\
                "Test CRP size: " + str(test_set_size) + "\n" +\
                "Test Accuracy: " + str(acc) + "\n" +\
                "LR: " + str(learning_rate) + "\n" +\
                "Number of attempts of re-training: " + str(attempt) +"\n" +\
                "Transfer Learning On: " + str(False)
                
        with open(prob_model_instance_dir + "/stats.txt",'w') as stat_file:
            stat_file.write(stats)

        return acc            
    ##___________________________________________________


##Sample Program logic:

userID = input("Enter UserID: ")
print("User with ID=" + userID +" is created.")
exec_program = TTP_Client                             (myID = userID,
                                                       broker_ip = 'localhost',
                                                       broker_port = 1883,
                                                       introduction_topic='client_introduction',
                                                       controller_executable_topic='controller_executable',
                                                       controller_echo_topic="echo",
                                                       start_loop=False)

while(True):
    output = exec_program.base_loop() # Restoration Cap is 10
    if(output == -1):
        print("User with ID=" + userID +" is re-created.")
        exec_program = TTP_Client                             (myID = userID,
                                                       broker_ip = 'localhost',
                                                       broker_port = 1883,
                                                       introduction_topic='client_introduction',
                                                       controller_executable_topic='controller_executable',
                                                       controller_echo_topic="echo",
                                                       start_loop=False)

        exec_program.echo_msg("Client " + exec_program.id + " is re-initialized...")
    

##____________________