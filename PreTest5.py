import json
import matplotlib.pyplot as plt

def analytical_formula(p:float):
    Pre = p**2*(1-p)*(5*p*(1-p)**2+2*(1-p)**3+2*(1-p)*p**2+p**4+4*(1-p)**2*p**2)
    return Pre

file_name = "test5.json"
with open(file_name) as config_file:
    config = json.load(config_file)

# test = config['test']
# sub_test = {}
# for ber in test.keys():
#     test[ber]=analytical_formula(float(ber))

test = {}
flag = True
ber = config['starting_ber']
while flag == True:
    test[ber]=analytical_formula(float(ber))
    ber = ber - ber/4
    if(ber < 1e-10):
        flag = False

# Save output data structures in file json
json_array = {
    'test': test,
}
array_to_file = json.dumps(json_array, indent=4)
filename = "ResultTest5.json"
with open(filename, "w") as outfile:
    outfile.write(array_to_file)

# print(type(test))
# print(len(test))
# for key in test:
#     ber = float(key)
#     if(ber>= 1e-3 and ber <= 0.5):
#         sub_test[ber] = float(test[key])
# 
# print(len(sub_test))
# print(sub_test)