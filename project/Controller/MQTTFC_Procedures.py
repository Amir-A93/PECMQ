import paho.mqtt.client as mqtt
import time as T
import os
class MQTTFC_Procedures:

    def __init__(self):
        self.wait_time = 5 # in sec
        self.registered_procedures = ['Topic_Naming_PUF_SinglePair','Topic_Naming_PUF_FivePair']
    
    def parse_procedure_command(self,command, cmdline,fc_parser_func):
        if(command in self.registered_procedures):
            if(command == 'Topic_Naming_PUF_SinglePair'):

                id1 = cmdline.split(' -id1 ')[1].split(' -id2 ')[0]
                id2 = cmdline.split(' -id2 ')[1].split(';')[0]
                self.Topic_Naming_PUF_SinglePair(id1,id2,fc_parser_func)

            if(command == 'Topic_Naming_PUF_FivePair'):

                id1 = cmdline.split(' -id1 ')[1].split(' -id2 ')[0]
                id2 = cmdline.split(' -id2 ')[1].split(' -id3 ')[0]
                id3 = cmdline.split(' -id3 ')[1].split(' -id4 ')[0]
                id4 = cmdline.split(' -id4 ')[1].split(' -id5 ')[0]
                id5 = cmdline.split(' -id5 ')[1].split(' -id6 ')[0]
                id6 = cmdline.split(' -id6 ')[1].split(' -id7 ')[0]
                id7 = cmdline.split(' -id7 ')[1].split(' -id8 ')[0]
                id8 = cmdline.split(' -id8 ')[1].split(' -id9 ')[0]
                id9 = cmdline.split(' -id9 ')[1].split(' -id10 ')[0]
                id10 = cmdline.split(' -id10 ')[1].split(';')[0]
                self.Topic_Naming_PUF_FivePair(id1,id2,id3,id4,id5,id6,id7,id8,id9,id10,fc_parser_func)



#runp Topic_Naming_PUF_FivePair -id1 KH_c1 -id2 KH_c6 -id3 KH_c2 -id4 KH_c7 -id5 KH_c3 -id6 KH_c8 -id7 KH_c4 -id8 KH_c9 -id9 KH_c5 -id10 KH_c10
    def Topic_Naming_PUF_SinglePair(self,id1,id2, parser_func):
        
        rep = 10
        topic_sizes = [16,32,64,96,128,192]
        code_length = [15]
        
        #topic_sizes = [16,32,64,96,128,192]
        #code_length = [1,2,3,4,5,6,7,8,9,10]
        #code_length = [15]
        #topic_sizes = [32]
        #code_length = [5]
        # topic_sizes = [16]
        # code_length = [4]
        
        for ts in topic_sizes:
            for cl in code_length:
                for i in range(rep):
                    parser_func("run establish_private_topic -id1 " + str(id1) + " -id2 " + str(id2) + " -l " + str(ts) + " -rl " + str(cl))
                    T.sleep(self.wait_time)
                    parser_func("run wisper -id1 " + str(id1) + " -id2 " + str(id2))
                    T.sleep(self.wait_time)
    

    def Topic_Naming_PUF_FivePair(self,id1,id2,id3,id4,id5,id6,id7,id8,id9,id10, parser_func):
        
        rep = 10
        # topic_sizes = [16,32,48,64,80,96,112,128,144,160,176,192]
        topic_sizes = [16,32,64,96,128,192]
        # code_length = [1,2,3,4,5,6,7,8,9,10]
        
        # topic_sizes = [192]
        code_length = [9]
        
        for ts in topic_sizes:
            for cl in code_length:
                for i in range(rep):
                    parser_func("run establish_private_topic -id1 " + str(id1) + " -id2 " + str(id2) + " -l " + str(ts) + " -rl " + str(cl))
                    T.sleep(self.wait_time)
                    parser_func("run establish_private_topic -id1 " + str(id3) + " -id2 " + str(id4) + " -l " + str(ts) + " -rl " + str(cl))
                    T.sleep(self.wait_time)
                    parser_func("run establish_private_topic -id1 " + str(id5) + " -id2 " + str(id6) + " -l " + str(ts) + " -rl " + str(cl))
                    T.sleep(self.wait_time)
                    parser_func("run establish_private_topic -id1 " + str(id7) + " -id2 " + str(id8) + " -l " + str(ts) + " -rl " + str(cl))
                    T.sleep(self.wait_time)
                    parser_func("run establish_private_topic -id1 " + str(id9) + " -id2 " + str(id10) + " -l " + str(ts) + " -rl " + str(cl))
                    T.sleep(self.wait_time)
