import os
import time
import jsonlines, json
import multiprocessing

lama_rel_patterns = {}

with jsonlines.open('/home/chandan/NLP/project/knowledge_neurons/data/LAMA/raw_data/relations.jsonl', 'r') as r_reader:
    for pattern in r_reader:
        lama_rel_patterns[pattern['relation']] = pattern

relations = list(lama_rel_patterns.keys())
num_relations = len(relations)//5
relations1 = relations[:num_relations]
relations2 = relations[num_relations:2*num_relations]
relations3 = relations[2*num_relations:3*num_relations]
relations4 = relations[3*num_relations:4*num_relations]
relations5 = relations[4*num_relations:]

def execute_1_run_mlm(relation_list):
    for key in relation_list:
        st=time.time()
        print('relation:',key)
        os.system('bash 1_run_mlm.sh '+key)
        et=time.time()
        print('time taken:',et-st,' s')
        print('\n')

process1 = multiprocessing.Process(target=execute_1_run_mlm, args=(relations1,))
process1.start()

process2 = multiprocessing.Process(target=execute_1_run_mlm, args=(relations2,))
process2.start()

process3 = multiprocessing.Process(target=execute_1_run_mlm, args=(relations3,))
process3.start()

process4 = multiprocessing.Process(target=execute_1_run_mlm, args=(relations4,))
process4.start()

process5 = multiprocessing.Process(target=execute_1_run_mlm, args=(relations5,))
process5.start()

process1.join()
process2.join()
process3.join()
process4.join()
process5.join()