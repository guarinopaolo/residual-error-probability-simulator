import json
import matplotlib.pyplot as plt

# file_name = "Results/Test4/prova1/Results_2024_04_16_11_55.json"
# file_name = "Results/Test4/prova2/Results_2024_04_19_12_15.json"
file_name = "test3.json"

with open(file_name) as config_file:
    config = json.load(config_file)

sub_test = {}
test = config['test']
print(type(test))
print(len(test))
for key in test:
    ber = float(key)
    if(ber>= 1e-3 and ber <= 0.5):
        sub_test[ber] = float(test[key])

print(len(sub_test))
print(sub_test)
# x_values = config['x_data_points_mc']
# x_values = [float(x) for x in x_values]
# y_values = config['y_data_points_mc']
# y_values = [float(y) for y in y_values]
# maxPre_x = float(config['max_ber_MINE'])
# maxPre_y = float(y_values[x_values.index(maxPre_x)])
# bounded_x_values = [num for num in x_values if '0.00' in str(num)]
# bounded_y_values = []
# for i in bounded_x_values:
#     bounded_y_values.append(y_values[x_values.index(i)])
# 
# plt.figure(figsize=(15, 8))
# plt.plot(x_values, y_values, label='Monte Carlo Simulation')
# for x, y in zip(bounded_x_values, bounded_y_values):
#     plt.scatter(x, y, color='green', label=f'iPoint: ({x}, {y})')
# plt.scatter(maxPre_x, maxPre_y, color='red', label=f'Maximum Point: ({maxPre_x}, {maxPre_y})')
# plt.xlabel('BER')
# plt.ylabel('Pre')
# plt.title('Monte Carlo Simulation')
# plt.legend()

#plt.savefig('Results/Test4/prova1/goodGraphs/MC.pdf')
# plt.savefig('Results/Test4/prova2/goodGraphs/MC.pdf')

# plt.show()