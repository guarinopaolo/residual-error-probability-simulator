import json

# file_name = "Results/Test4/prova1/Results_2024_04_16_11_55.json"
# file_name = "Results/Test4/prova2/Results_2024_04_19_12_15.json"
file_name = "test1.json"

with open(file_name) as config_file:
    config = json.load(config_file)


binary_string = config['original_sample']

# Convert binary string to hexadecimal string using f-string
hexadecimal_string = f"{int(binary_string, 2):X}"

print(hexadecimal_string)  

# Convert hexadecimal string to binary string using f-string
binary_string = f"{int(hexadecimal_string, 16):b}"

while(len(binary_string)!=len(config['original_sample'])):
    binary_string = "0" + binary_string

print(f"Converted string:\n{binary_string}\nOriginal sample: \n{config['original_sample']}")
print(len(config['original_sample']))
print(len(binary_string))

f = config['original_sample']
for i in range(0, len(binary_string)):
    if(binary_string[i]!=f[i]):
        print(f"Error found at {i}: {binary_string[i]}!={f[i]}")