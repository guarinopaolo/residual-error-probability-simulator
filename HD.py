import os
import json

#directory = "Results/Test1/prova2/UndetectedErrorMasks"
#directory = "Results/Test3/prova4/UndetectedErrorMasks"
#directory = "Results/Test5/prova3/UndetectedErrorMasks"
directory = "Results/Test6/prova2/UndetectedErrorMasks"

file_hd_estimation = {}
undetected_frames = {}

def open_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        data = open_file(filepath)
        undetected_frames = data['undetected_frames']
        list = []
        for key in undetected_frames.keys():
            list.append(key.count('1'))
        file_hd_estimation[filepath]=list

hd = float('inf')
for key in file_hd_estimation.keys():
    min_bits_sum = min(file_hd_estimation[key])
    if(min_bits_sum<hd):
        hd = min_bits_sum

print(f"The estimated Hamming distance is: {hd}.")

# Save output data structures in file json
json_array = {
    'HD_estimated':hd,
    'HD_estimation_per_BER':file_hd_estimation
}
array_to_file = json.dumps(json_array, indent=4)

# directory = "Results/HD/Test1"
# directory = "Results/HD/Test3"
# directory = "Results/HD/Test5"
directory = "Results/HD/Test6"

filename = "HD.json"
if not os.path.exists(directory):
    os.makedirs(directory)
filepath = os.path.join(directory, filename)
with open(filepath, "w") as outfile:
    outfile.write(array_to_file)