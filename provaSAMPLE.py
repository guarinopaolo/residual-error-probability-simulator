import json
import numpy as np

# file_name = "Results/Test4/prova1/Results_2024_04_16_11_55.json"
# file_name = "Results/Test4/prova2/Results_2024_04_19_12_15.json"
file_name = "test1.json"

with open(file_name) as config_file:
    config = json.load(config_file)

original_sample = (np.frombuffer(buffer=config['original_sample'].encode('utf-8'), dtype='int8').copy()) - 48

print(config['original_sample'])
print(type(config['original_sample']))
print(original_sample)
print(type(original_sample))

s = config['original_sample']
for i in range(0,len(original_sample)):
    if(s[i] != str(original_sample[i])):
        print(f'Found different value at {i}: {s[i]} != {original_sample[i]}')

print(len(s))
print(len(original_sample))