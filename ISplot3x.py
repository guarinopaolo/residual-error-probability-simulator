import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum

def importance_sampling(config: dict, is_dict: dict, q: float):
    probs = []
    test = config['test']
    for key in test.keys():
        probs.append(float(key))
    probs.sort()
    is_dict = {float(key): value for key, value in is_dict.items()}
    estimator_sum = 0
    ListSumBitsErrorMask = is_dict[q]
    is_values = {}
    itrs = float(config['iterations'])
    
    for p in probs:
        if(p <= q):
            for i in ListSumBitsErrorMask:
                # formula 5.1 and 5.2
                # estimator_sum += pow((p / q), i) * pow(((1 - p) / (1 - q)), (int(len(self.config['original_sample'])) - i))
                estimator_sum += pow((p / q), i) * pow(((1 - p) / (1 - q)), (int(len(config['original_sample'])) - i))
            # theta (5.2), estimation of Pre
            estimated_prob = float(estimator_sum) / float(config['iterations'])
            # print(f'estimated_prob({p}) = {float(estimator_sum)}/{itrs} = {estimated_prob}')
            is_values[p] = estimated_prob
            estimator_sum = 0
    return is_values

def plot_mc_and_is(pre_dict: dict, is_xy: dict, q: float, max_pre_x_mc: float):
        pre_dict = {k: pre_dict[k] for k in sorted(pre_dict.keys(), key=float)}
        is_xy = {k: is_xy[k] for k in sorted(is_xy.keys(), key=float)}
        x_values_mc = list(pre_dict.keys())
        x_values_mc = [float(element) for element in x_values_mc]
        y_values_mc = list(pre_dict.values())
        y_values_mc = [float(element) for element in y_values_mc]
        x_values_is = list(is_xy.keys())
        x_values_is = [float(element) for element in x_values_is]
        y_values_is = list(is_xy.values())
        y_values_is = [float(element) for element in y_values_is]
        Pre_q = float(y_values_mc[x_values_mc.index(q)])

        plt.figure(figsize=(15, 8))
        if(q==max_pre_x_mc):
            plt.scatter(q, Pre_q, color='red', label=f'Maximum Point: ({round(q, 5)}, {Pre_q})')
        else:
            plt.scatter(q, Pre_q, color='green', label=f'Point: ({round(q, 5)}, {Pre_q})')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Bit Error Rate (BER)')
        plt.ylabel('Residual Error Probability (Pre)')
        #plt.title('Monte Carlo Simulation')
        plt.grid(True)
        plt.plot(x_values_is, y_values_is, color='orange', label='Importance Sampling')
        plt.plot(x_values_mc, y_values_mc, color='blue', label='Monte Carlo Simulation')
        
        is_y_values_upper = []
        for i in x_values_mc:
            if(float(i)>=1e-3 and float(i)<= 0.5):
                is_y_values_upper.append(is_xy[i])
        
        # Mean Squared Error (MSE) between Monte Carlo and Importance Sampling for values in [10^-3;0.5]
        mse_upper_mc_is = np.mean((np.array(y_values_mc) - np.array(is_y_values_upper))**2)
        # Calculate Pearson correlation coefficient between Monte Carlo and Importance Sampling for values in [10^-3;0.5]
        # pearson_corr_upper_mc_is = np.corrcoef(y_values_mc, is_y_values_upper)[0, 1]
        
        # Create first legend
        legend1 = plt.legend(loc='upper left')
        # Plotting a dummy point for the second legend
        plt.plot([], [], ' ', label="Extra Legend")
        # legend2 = plt.legend([f'MC-IS for BER in [10^-3;0.5], MSE: {mse_upper_mc_is}, Pearson: {pearson_corr_upper_mc_is}'], loc='lower right')
        legend2 = plt.legend([f'MC-IS for BER in [10^-3;0.5], MSE: {mse_upper_mc_is}'], loc='lower right')
        # Add both legends to the plot
        plt.gca().add_artist(legend1)
        
        # directory = "Results/CAN-FD3/prova1/Graphs"
        # directory = "Results/CAN-FD1/prova2/Graphs"
        directory = "Results/CAN-FD/Graphs"
        filename = f"MC_and_IS_{q}.pdf"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, filename))
        plt.close()

def plot_mc_and_is_overlapped(pre_dict1: dict, pre_dict2: dict, pre_dict3: dict,
                              is_xy1: dict, is_xy2: dict, is_xy3: dict,
                              q1: float, q2: float, q3: float,
                              max_pre_x_mc1: float, max_pre_x_mc2: float, max_pre_x_mc3: float,
                            ):
        
        pre_dict1 = {k: pre_dict1[k] for k in sorted(pre_dict1.keys(), key=float)}
        is_xy1 = {k: is_xy1[k] for k in sorted(is_xy1.keys(), key=float)}
        x_values_mc1 = list(pre_dict1.keys())
        x_values_mc1 = [float(element) for element in x_values_mc1]
        y_values_mc1 = list(pre_dict1.values())
        y_values_mc1 = [float(element) for element in y_values_mc1]
        x_values_is1 = list(is_xy1.keys())
        x_values_is1 = [float(element) for element in x_values_is1]
        y_values_is1 = list(is_xy1.values())
        y_values_is1 = [float(element) for element in y_values_is1]
        Pre_q1 = float(y_values_mc1[x_values_mc1.index(q1)])

        pre_dict2 = {k: pre_dict2[k] for k in sorted(pre_dict2.keys(), key=float)}
        is_xy2 = {k: is_xy2[k] for k in sorted(is_xy2.keys(), key=float)}
        x_values_mc2 = list(pre_dict2.keys())
        x_values_mc2 = [float(element) for element in x_values_mc2]
        y_values_mc2 = list(pre_dict2.values())
        y_values_mc2 = [float(element) for element in y_values_mc2]
        x_values_is2 = list(is_xy2.keys())
        x_values_is2 = [float(element) for element in x_values_is2]
        y_values_is2 = list(is_xy2.values())
        y_values_is2 = [float(element) for element in y_values_is2]
        Pre_q2 = float(y_values_mc2[x_values_mc2.index(q2)])

        pre_dict3 = {k: pre_dict3[k] for k in sorted(pre_dict3.keys(), key=float)}
        is_xy3 = {k: is_xy3[k] for k in sorted(is_xy3.keys(), key=float)}
        x_values_mc3 = list(pre_dict3.keys())
        x_values_mc3 = [float(element) for element in x_values_mc3]
        y_values_mc3 = list(pre_dict3.values())
        y_values_mc3 = [float(element) for element in y_values_mc3]
        x_values_is3 = list(is_xy3.keys())
        x_values_is3 = [float(element) for element in x_values_is3]
        y_values_is3 = list(is_xy3.values())
        y_values_is3 = [float(element) for element in y_values_is3]
        Pre_q3 = float(y_values_mc3[x_values_mc3.index(q3)])

        plt.figure(figsize=(15, 8))
        # plt.figure(figsize=(10, 8))
        if(q1==max_pre_x_mc1):
            plt.scatter(q1, Pre_q1, color='red', label=f'Maximum Point: ({round(q1, 5)}, {Pre_q1})')
        else:
            plt.scatter(q1, Pre_q1, color='green', label=f'Point: ({round(q1, 5)}, {Pre_q1})')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Bit Error Rate (BER)')
        plt.ylabel('Residual Error Probability (Pre)')
        #plt.title('Monte Carlo Simulation')
        plt.grid(True)
        plt.plot(x_values_is1, y_values_is1, color='orange', linewidth=2, label=f'Importance Sampling q={round(q1, 5)}')
        plt.plot(x_values_mc1, y_values_mc1, color='blue', linewidth=2, label='Monte Carlo Simulation')

        if(q2==max_pre_x_mc2):
            plt.scatter(q2, Pre_q2, color='pink', label=f'Maximum Point: ({round(q2, 5)}, {Pre_q2})')
        else:
            plt.scatter(q2, Pre_q2, color='brown', label=f'Point: ({round(q2, 5)}, {Pre_q2})')
        plt.plot(x_values_is2, y_values_is2, color='orange', linestyle='--', linewidth=2, label=f'Importance Sampling q={round(q2, 5)}')
        plt.plot(x_values_mc2, y_values_mc2, color='blue', linestyle='--', linewidth=2, label='Monte Carlo Simulation')
        
        if(q3==max_pre_x_mc3):
            plt.scatter(q3, Pre_q3, color='violet', label=f'Maximum Point: ({round(q3, 5)}, {Pre_q3})')
        else:
            plt.scatter(q3, Pre_q3, color='black', label=f'Point: ({round(q3, 5)}, {Pre_q3})')
        plt.plot(x_values_is3, y_values_is3, color='orange', linestyle=':', linewidth=2, label=f'Importance Sampling q={round(q3, 5)}')
        plt.plot(x_values_mc3, y_values_mc3, color='blue', linestyle=':', linewidth=2, label='Monte Carlo Simulation')
        
        plt.legend(loc='lower right')
        
        directory = "Results/CAN-FD/Graphs3x"
        filename = "MC_and_IS.pdf"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, filename),bbox_inches='tight', pad_inches=0)
        plt.close()

output_name1 = "Results/CAN-FD1/prova2/Results_2024_05_24_00_00.json"
with open(output_name1) as config_file1:
    output1 = json.load(config_file1)

output_name2 = "Results/CAN-FD2/prova1/Results_2024_05_24_22_29.json"
with open(output_name2) as config_file2:
    output2 = json.load(config_file2)

output_name3 = "Results/CAN-FD3/prova1/Results_2024_05_23_09_44.json"
with open(output_name3) as config_file3:
    output3 = json.load(config_file3)

pre_dict1 = output1['pre_dict']
is_dict1 = output1['is_dict']
maxPre_x1 = float(output1['max_ber'])

pre_dict2 = output2['pre_dict']
is_dict2 = output2['is_dict']
maxPre_x2 = float(output2['max_ber'])

pre_dict3 = output3['pre_dict']
is_dict3 = output3['is_dict']
maxPre_x3 = float(output3['max_ber'])

conf_name3 = "CAN-FD3.json"
conf_name2 = "CAN-FD2.json"
conf_name1 = "CAN-FD1.json"

with open(conf_name3) as config_file3:
        config3 = json.load(config_file3)
with open(conf_name2) as config_file2:
        config2 = json.load(config_file2)
with open(conf_name1) as config_file1:
        config1 = json.load(config_file1)

q3 = "0.0015856059695"
q2 = "0.0066817305051"
# q2 = "0.0066817305051"
q1 = "0.0089089740068"

is_xy1 = {}
is_xy2 = {}
is_xy3 = {}

is_xy1 = importance_sampling(config1, is_dict1, float(q1))
is_xy2 = importance_sampling(config2, is_dict2, float(q2))
is_xy3 = importance_sampling(config3, is_dict3, float(q3))

plot_mc_and_is_overlapped(pre_dict1, pre_dict2, pre_dict3, 
                          is_xy1, is_xy2, is_xy3,
                          float(q1), float(q2), float(q3),
                          maxPre_x1, maxPre_x2, maxPre_x3)