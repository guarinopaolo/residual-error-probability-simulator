import json
import matplotlib.pyplot as plt

# file_name = "Results/Test4/prova1/Results_2024_04_16_11_55.json"
# file_name = "Results/Test4/prova2/Results_2024_04_19_12_15.json"
file_name = "NewResults/Test3/prova3/Results_2024_04_28_19_33.json"

with open(file_name) as config_file:
    config = json.load(config_file)

# Unzip data points for plotting
pre_dict = {
        0.0008919033578: 1e-06,
        0.0010000000000: 0,
        0.0011892044771: 1e-06,
        0.0015856059695: 0.0,
        0.0021141412926: 1e-06,
        0.0025000000000: 3e-06,
        0.0028188550568: 0.0,
        0.0037584734091: 4e-06,
        0.0050000000000: 8e-06,
        0.0050112978788: 1.3e-05,
        0.0066817305051: 1e-05,
        0.0075000000000: 25e-06,
        0.0089089740068: 2.4e-05,
        0.0100000000000: 22e-06,
        0.0118786320090: 2.9e-05,
        0.0158381760120: 4.4e-05,
        0.0211175680161: 4.4e-05,
        0.0250000000000: 45e-06,
        0.0281567573547: 3.8e-05,
        0.0375423431396: 4.5e-05,
        0.0500000000000: 32e-06,
        0.0500564575195: 3.2e-05,
        0.0667419433594: 2e-05,
        0.0750000000000: 16e-06,
        0.0889892578125: 1.3e-05,
        0.1000000000000: 19e-06,
        0.1186523437500: 1.9e-05,
        0.1582031250000: 1.8e-05,
        0.2109375000000: 1.6e-05,
        0.2500000000000: 16e-06,
        0.2812500000000: 1.1e-05,
        0.3750000000000: 1.1e-05,
        0.5000000000000: 1.9e-05
}
x_values_mc = list(pre_dict.keys())
y_values_mc = list(pre_dict.values())
maxPre_x = float(0.0375423431396)
maxPre_y = float(y_values_mc[x_values_mc.index(maxPre_x)])
bounded_x_values = [num for num in x_values_mc if '0.00' in str(num)]
bounded_y_values = []
for i in bounded_x_values:
    bounded_y_values.append(y_values_mc[x_values_mc.index(i)])
# Remove the points having 0 as y-coordinate (i.e., Pre=0)
# Sort points based on x-coordinate
bounded_values = list(zip(bounded_x_values, bounded_y_values))
print(bounded_x_values)
print(bounded_y_values)
bounded_values = [(x_val, y_val) for x_val, y_val in zip(bounded_x_values, bounded_y_values) if y_val != 0]
bounded_x_values = [point[0] for point in bounded_values]
bounded_y_values = [point[1] for point in bounded_values]
print(bounded_x_values)
print(bounded_y_values)

plt.figure(figsize=(15, 8))
for x, y in zip(bounded_x_values, bounded_y_values):
    plt.scatter(x, y, color='green', label=f'Point: ({round(x, 5)}, {y})')
plt.yscale('log')
plt.xscale('log')
plt.scatter(maxPre_x, maxPre_y, color='red', label=f'Maximum Point: ({round(maxPre_x,5)}, {maxPre_y})')
plt.xlabel('Bit Error Rate (BER)')
plt.ylabel('Residual Error Probability (Pre)')
plt.title('Monte Carlo Simulation')
plt.grid(True)
plt.plot(x_values_mc, y_values_mc, color='blue', label='Monte Carlo Simulation')
plt.legend()
#plt.show()
filename = 'NewResults/Test3/prova3/goodGraphs2/MC.pdf'
plt.savefig(filename)