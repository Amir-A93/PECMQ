from executable_class import *
import numpy as np
from ruhrmair_python.PUFmodels import XORArbPUF, linArbPUF
import os
from os import path
import pickle
import pandas as pd
import json
import hashlib
import datetime
import galois
from reedsolo import RSCodec
import tracemalloc
import time
import ldp_simulator_new as ldp

class PUF_Client(PubSub_Base_Executable): ##CHANGE:: change class name
    
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
                 start_loop,
                 challenge_size,
                 xor_size):
        
        
        ## Here you initialize the class specific variables
        self.challenge_size = challenge_size
        self.xor_size = xor_size
       
        self.list_of_partners = [] #[partner_id, topic]

        self.code = None
        
        ##_________________________________________________

        ##IMPORTANT:: This line of code is needed to append the name of newly defined class specific executables::
        ## CHANGE:: --> self.executables.append([<<list of executables>>>])
        self.executables.append("wisper")
        self.executables.append("wisper_log")
        self.executables.append("enroll")
        self.executables.append("challenge")
        self.executables.append("register_private_topic")
        self.executables.append("subscribe_to_private_topic")
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

        self.PUF_instance = self.Load_PUF()
        self.client_to_TTP_topic = "C2TTP"
        self.TTP_to_client_topic = "TTP2C"
        self.Private_TTP_to_Client_topic = "TTP2"+ str(self.id)
        self.Private_Client_to_TTP_topic =  str(self.id)+"2TTP"
        self.client.subscribe(self.TTP_to_client_topic)
        self.client.subscribe(self.Private_TTP_to_Client_topic)
       
        ##____________________________________________________________________________________________________________________________________________________

    def execute_on_msg (self,client,userdata, msg):
        PubSub_Base_Executable.execute_on_msg(self,client,userdata,msg)

        # try:
        header_body = str(msg.payload.decode()).split('::')
        header_parts = header_body[0].split('|')
        
            ##IMPORTANT:: Here you extend the message parser to check for the class specific executables
            ## CHANGE:: --> if msg_parts[0] == <<executable function name>>:
                ##Here you execute accordingly, or simply just invoke the executable: example: self.<<executable name>>(parameterst)
                ##__________________________________________________________________________________________________________________
            
        if(not(header_parts[2] in self.executables)):
            self.ERROR_executable_not_defined(msg.payload.decode())
            return

        if header_parts[2] == 'challenge':
            challenge = header_body[1].split('-c ')[1].split(';')[0]
            self.challenge(challenge)

        if header_parts[2] == 'wisper':
            id1 = header_body[1].split('-id1 ')[1].split(' -id2 ')[0]
            id2 = header_body[1].split(' -id2 ')[1].split(';')[0]
            self.wisper(id1,id2)

        if header_parts[2] == 'wisper_log':
            client_id = header_body[1].split('-c ')[1].split(' -rl ')[0]
            msg = str(msg.payload.decode())
            self.wisper_log(msg,client_id)

        if header_parts[2] == 'enroll':
            
            size = header_body[1].split('-s ')[1].split(' -id ')[0]
            id = header_body[1].split(' -id ')[1].split(';')[0]
            if((id == self.id) or id == 'all'):
                self.enroll_with_ldp(size)
            else:
                print("Enroll call is not mine.")

        if header_parts[2] == 'register_private_topic':
            id = header_body[1].split(' -id ')[1].split(' -c ')[0]
            partner_id = header_body[1].split(' -c ')[1].split(' -rl ')[0]
            replen = int(header_body[1].split(' -rl ')[1].split(' -cset ')[0])
            cset = header_body[1].split('-cset ')[1].split(';')[0]
            self.register_private_topic(id,cset,pid = partner_id,repetition_length = replen)    

        if header_parts[2] == 'subscribe_to_private_topic':
            id = header_body[1].split('-id ')[1].split(' -rl ')[0]
            replen = int(header_body[1].split(' -rl ')[1].split(' -cset ')[0])
            topic = header_body[1].split(' -cset ')[1].split(';')[0]
            self.subscribe_to_private_topic(id,topic, repetition_length = replen)

        # except Exception as e:
        #     print("Error occured or message was not right!")
        #     print("ERROR: " + str(e))
        ##________________________________________________________________________________

    ##Here you define the new class specific executables:
    def challenge(self, challenge_vector):
        print(self.id + "Challenging PUF with " + challenge_vector)

    def enroll_with_ldp(self, set_size):
        print(self.id + " Enrolling to TTP")
        size = int(set_size)
        set_dict = {}
        challenge_set   = self.PUF_instance.generate_challenge(size)
        feature_set     = self.PUF_instance.calc_features(challenge_set)
        xor_responses   = self.PUF_instance.bin_response(feature_set)
        xor_responses           = np.array([(int(1-i)/2) for i in xor_responses])
        new_feature_set    = np.transpose(feature_set,(2,0,1))
        # new_feature_set    = new_feature_set[:,0,:self.challenge_size+1]
        
        new_challenge_set = np.transpose(challenge_set,(2,0,1))
        # new_challenge_set = new_challenge_set[:,0,:self.challenge_size]
        print("new challenge set shape: " + str(new_challenge_set.shape))
        for i in range(size):
            set_dict [str(i)] = {'challenge': new_challenge_set[i,0,:].tolist(),
                                 'feature'  : new_feature_set[i,0,:].tolist(),
                                 'response' : xor_responses[i]}

        puf_dir =  "CLIENT_PUF_" + self.id
        set_json_file = open(puf_dir+"/original_crp_"+str(size)+".json",'w')
        json.dump(set_dict,set_json_file)
        set_json_file.close()

        segmented_input_json_file, output_json_file  = ldp.process_json(puf_dir+"/original_crp_"+str(size)+".json", 
                                                                        puf_dir+"/original_segmented_crp_"+str(size)+".json",
                                                                        puf_dir+"/ldp_crp_"+str(size)+".json")

        ldp_crp_set = ''
        outfile = open(output_json_file, 'r')
        ldp_set = json.load(outfile)
        group_counts = ldp.count_response_bits(ldp_set,False,"")
        for key in ldp_set:
            final_val = 0
            if (group_counts[key]['ones'] >= group_counts[key]['zeros']):
                final_val = 1
            ldp_set[key]['avg_response'] = final_val 
        ldp_crp_set = json.dumps(ldp_set)
        outfile.close()

        self.publish(self.client_to_TTP_topic, 'enroll_device_ldp',"-id " + str(self.id) + " -csize " + str(self.challenge_size) + " -nxor " + str(self.xor_size) + " -crp " + ldp_crp_set)
        


    def enroll_legacy(self,set_size):
        print(self.id + " Enrolling to TTP")
        size = int(set_size)
        challenge_set   = self.PUF_instance.generate_challenge(size)
        feature_set     = self.PUF_instance.calc_features(challenge_set)
        xor_responses   = self.PUF_instance.bin_response(feature_set)
        #responses       = self.PUF_instance.indiv_responses(feature_set)

        # for i_1 in range(responses.shape[0]):
        #     for j_1 in range(responses.shape[1]):
        #         responses[i_1,j_1] = (int(1-responses[i_1,j_1])/2)
        
        xor_responses           = np.array([(int(1-i)/2) for i in xor_responses])
        training_feature_set    = np.transpose(feature_set,(2,0,1))
        training_feature_set    = training_feature_set[:,0,:self.challenge_size+1]
        
        training_challenge_set = np.transpose(challenge_set,(2,0,1))
        training_challenge_set = training_challenge_set[:,0,:self.challenge_size]


        challenge_flattened = np.array(training_feature_set).flatten().astype(int)
        challenge_str =''
        for ci in challenge_flattened:
            challenge_str = challenge_str + str(ci) + ' '
        challenge_str = challenge_str[0:len(challenge_str)-1]

        resp_flattened = np.array(xor_responses).flatten().astype(int)
        resp_str =''
        for ri in resp_flattened:
            resp_str = resp_str + str(ri) + ' '
        resp_str = resp_str[0:len(resp_str)-1]

        meta_str = str(feature_set.shape[0]) + ' ' + str(training_feature_set.shape[0]) + ' ' + str(training_feature_set.shape[1])
        crp_str = meta_str + "&" + challenge_str + "&" + resp_str
       
        # print(xor_responses.shape)
        # Send the CRP set to TTP
        self.publish(self.client_to_TTP_topic, 'enroll_device',"-id " + str(self.id) + " -csize " + str(self.challenge_size) + " -nxor " + str(self.xor_size) + " -crp " + crp_str)
        
    
    def wisper(self, myid, partner_id):
        if(myid == self.id):
            partner_found = 0
            for partner in self.list_of_partners:
                if(partner_id == partner[0]):
                    self.publish(partner[1],'wisper_log'," -c " + self.id + " -rl " + str(partner[2]))
                    print(self.id + " Wispered to " + partner_id)
                    partner_found = 1
                    break
            if(partner_found == 0):
                print("Partner not found!")
        else:
            print("Wisper request is not mine!")
        
    def wisper_log(self,msg,client):
        wisper_log_dir =  "CLIENT_PUF_" + self.id + "/wisper_logs/"
        if(path.exists(wisper_log_dir) != True):
            os.makedirs(wisper_log_dir)
        
        with open(wisper_log_dir + "/" + str(client) + "_wisper_log.txt", 'a') as output:
            output.write(msg + "\n")  




    def register_private_topic(self,id,cset_cipher,pid,repetition_length):
        
        cpu_start_time = time.process_time()
        # Start measuring memory allocation
        tracemalloc.start()

        cset = self.decrypt_msg(str(cset_cipher).encode())
        if(id == self.id):
            challenge_set = json.loads(cset)
            challenge_set = np.transpose(challenge_set,(1,0))
            
          
            new_challenge_set = self.concat_challenge(challenge_set)
            feature_set = self.PUF_instance.calc_features(new_challenge_set)
            
            puf_infer_before = datetime.datetime.now()
            uts_1 = datetime.datetime.timestamp(puf_infer_before)*1000
            responses = self.PUF_instance.bin_response(feature_set)
            puf_infer_after = datetime.datetime.now()
            uts_2 = datetime.datetime.timestamp(puf_infer_after)*1000
            puf_inference_tdelta = uts_2 - uts_1

            topic = self.decode_topic(repetition_length,responses)
           
            #print(self.list_of_partners)
            partner_index = -1
            for p in range(len(self.list_of_partners)):
                if self.list_of_partners[p][0] == pid:
                    partner_index = p
            if(partner_index >= 0):

                hashed_topic = hashlib.sha256(topic.encode()).hexdigest()
                
                self.list_of_partners[partner_index][1] = hashed_topic
                self.list_of_partners[partner_index][2] = int(repetition_length)

                transmission_init_time = datetime.datetime.now()
                Publish_time = datetime.datetime.timestamp(transmission_init_time)*1000
        
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                # Stop measuring CPU time
                cpu_end_time = time.process_time()
                memory_usage = current / 10**6  # Convert to MB
                peak_memory_usage = peak / 10**6  # Convert to MB
                cpu_time_used = cpu_end_time - cpu_start_time  # In seconds

                self.publish(self.client_to_TTP_topic, 'verify_hash',   "-id " + str(self.id) + 
                                                                        " -hash " + str(hashed_topic) + 
                                                                        " -tpuf " + str(puf_inference_tdelta) + 
                                                                        " -pubtime " + str(Publish_time) + 
                                                                        " -memavg " + str(memory_usage) + 
                                                                        " -mempeak " + str(peak_memory_usage) + 
                                                                        " -cput " + str(cpu_time_used))
        
                print("Updated topic " + topic)
            else:
                hashed_topic = hashlib.sha256(topic.encode()).hexdigest()
              
                self.list_of_partners.append([pid,hashed_topic,repetition_length])
                
                transmission_init_time = datetime.datetime.now()
                Publish_time = datetime.datetime.timestamp(transmission_init_time)*1000
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                # Stop measuring CPU time
                cpu_end_time = time.process_time()
                memory_usage = current / 10**6  # Convert to MB
                peak_memory_usage = peak / 10**6  # Convert to MB
                cpu_time_used = cpu_end_time - cpu_start_time  # In seconds

                self.publish(self.client_to_TTP_topic, 'verify_hash',   "-id " + str(self.id) + 
                                                                        " -hash " + str(hashed_topic) + 
                                                                        " -tpuf " + str(puf_inference_tdelta) + 
                                                                        " -pubtime " + str(Publish_time) + 
                                                                        " -memavg " + str(memory_usage) + 
                                                                        " -mempeak " + str(peak_memory_usage) + 
                                                                        " -cput " + str(cpu_time_used))
        
                print("Registered topic " + hashed_topic)

            
            

    def concat_challenge(self,challenges):
            new_challenges = np.empty([self.xor_size, self.challenge_size, challenges.shape[1]])
            for puf in range(self.xor_size):
                new_challenges[puf] = challenges
            return new_challenges

    def subscribe_to_private_topic(self,id,cset_cipher,repetition_length):
        cpu_start_time = time.process_time()
        # Start measuring memory allocation
        tracemalloc.start()

        cset = self.decrypt_msg(str(cset_cipher).encode())
        if(str(id) == self.id):
            challenge_set = json.loads(cset)
            challenge_set = np.transpose(challenge_set,(1,0))
            
  
            new_challenge_set = self.concat_challenge(challenge_set)
            feature_set = self.PUF_instance.calc_features(new_challenge_set)

            puf_infer_before = datetime.datetime.now()
            uts_1 = datetime.datetime.timestamp(puf_infer_before)*1000
            responses = self.PUF_instance.bin_response(feature_set)
            puf_infer_after = datetime.datetime.now()
            uts_2 = datetime.datetime.timestamp(puf_infer_after)*1000
            puf_inference_tdelta = uts_2 - uts_1

            wisper_topic = self.decode_topic(repetition_length,responses)
            hashed_topic = hashlib.sha256(wisper_topic.encode()).hexdigest()
            self.client.subscribe(topic=hashed_topic)

            transmission_init_time = datetime.datetime.now()
            Publish_time = datetime.datetime.timestamp(transmission_init_time)*1000
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # Stop measuring CPU time
            cpu_end_time = time.process_time()
            memory_usage = current / 10**6  # Convert to MB
            peak_memory_usage = peak / 10**6  # Convert to MB
            cpu_time_used = cpu_end_time - cpu_start_time  # In seconds

            self.publish(self.client_to_TTP_topic, 'verify_hash',   "-id " + str(self.id) +
                                                                    " -hash " + str(hashed_topic) +
                                                                    " -tpuf " + str(puf_inference_tdelta) + 
                                                                    " -pubtime " + str(Publish_time) + 
                                                                    " -memavg " + str(memory_usage) + 
                                                                    " -mempeak " + str(peak_memory_usage) + 
                                                                    " -cput " + str(cpu_time_used))
            
            print("Subscribed to topic " + hashed_topic)
    ##___________________________________________________

    ##Here you define the rest of the class logic:

    def Create_PUF(self):
        puf_dir =  "CLIENT_PUF_" + self.id
        if(path.exists(puf_dir) != True):
            os.mkdir(puf_dir)
            
        
        puf_instance = XORArbPUF(num_bits=self.challenge_size,
                                      numXOR=self.xor_size,
                                      type='equal')
        
        with open(puf_dir + "/puf_model.pkl", 'wb') as output:
            pickle.dump(puf_instance, output, pickle.HIGHEST_PROTOCOL)
        
        return puf_instance
        
    def Load_PUF(self):
        puf_dir =  "CLIENT_PUF_" + self.id
        if(path.exists(puf_dir) != True):
            print("No PUF model director has been found. Creating a new PUF instance...")
            puf_instance = self.Create_PUF()
        else:        
            with open(puf_dir + "/puf_model.pkl", 'rb') as model_file:
                puf_instance = pickle.load(model_file)
            
        return puf_instance


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

    def decode_with_reed_solomon(self,encoded_binary_array, n_parity_symbols):
        
        # if(self.code == None):
        #     self.code = RSCodec(n_parity_symbols)
        # rsc = self.code
        rsc = RSCodec(n_parity_symbols)

        byte_data = self.binary_to_bytes(encoded_binary_array)
        decoded_data = rsc.decode(byte_data)
        decoded_binary_array = self.bytes_to_binary(list(decoded_data[0]))
        return decoded_binary_array

    def decode_bch(self,encoded_message,block_length):
        n = block_length
        k = 5  # Message length
        
        # if(self.code == None):
        #     self.code = galois.BCH(n, k)
        # bch = self.code
        bch = galois.BCH(n, k)

        msg =  encoded_message.reshape((int(len(encoded_message)/n),n))
        decoded_message = bch.decode(msg)
        return decoded_message.reshape(decoded_message.shape[0]*decoded_message.shape[1])

    def repetition_decode(self,encoded_array, n_repeats):
        reshaped_array = encoded_array.reshape(-1, n_repeats)
        decoded_array = np.sum(reshaped_array, axis=1) > (n_repeats // 2)
        return decoded_array.astype(int)


    # def decode_topic(self, rep_length, responses):
    #     print(responses)
    #     topic_size = int(len(responses) / rep_length)
    #     topic = ""
    #     th = rep_length / 2.0
    #     for i in range(0,len(responses),rep_length):
    #         one_sum = 0
    #         bit = 0
    #         for j in range(rep_length):
    #             resp_bit = 0
    #             if(responses[i + j] < 0):
    #                 resp_bit = 1

    #             one_sum += resp_bit
            
    #         if(one_sum > th):
    #             bit = 1
    #         topic = topic + str(bit)
    #     # topic_hash = hashlib.sha256(topic.encode('utf-8')).hexdigest()
        
    #     return topic
    
    def decode_topic(self, rep_length, responses):
        
        topic = ""
        for i in range(len(responses)):
            if(responses[i]<0):
                responses[i] = 1
            else:
                responses[i] = 0
        responses = np.array(responses,dtype=int)
        # print(responses)
        decoded_msg = self.decode_bch(responses,rep_length)
        for bit in decoded_msg:
            topic = topic + str(int(bit))
        # topic_hash = hashlib.sha256(topic.encode('utf-8')).hexdigest()
        
        return topic
    ##___________________________________________________


##Sample Program logic:

userID = input("Enter UserID: ")
print("User with ID=" + userID +" is created.")
exec_program = PUF_Client                             (myID = userID,
                                                       broker_ip = 'localhost',
                                                       broker_port = 1883,
                                                       introduction_topic='client_introduction',
                                                       controller_executable_topic='controller_executable',
                                                       controller_echo_topic="echo",
                                                       start_loop=False,
                                                       challenge_size=64,
                                                       xor_size = 2)



while(True):
    output = exec_program.base_loop() # Restoration Cap is 10
    if(output == -1):
        print("User with ID=" + userID +" is re-created.")
        exec_program = PUF_Client                             (myID = userID,
                                                       broker_ip = 'localhost',
                                                       broker_port = 1883,
                                                       introduction_topic='client_introduction',
                                                       controller_executable_topic='controller_executable',
                                                       controller_echo_topic="echo",
                                                       start_loop=False,
                                                       challenge_size=64,
                                                       xor_size = 2)

        exec_program.echo_msg("Client " + exec_program.id + " is re-initialized...")
    
##____________________