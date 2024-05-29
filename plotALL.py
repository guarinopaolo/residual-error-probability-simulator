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

def plot_mc_anltcl_is(pre_dict: dict, is_xy: dict, q: float, max_pre_x_mc: float, input: dict, r: int):
        test = input['test']
        test = {k: test[k] for k in sorted(test.keys(), key=float)}
        x_values_test = [float(key) for key in test.keys()]
        y_values_test = [float(test[key]) for key in test.keys()]

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
            plt.scatter(q, Pre_q, color='red', linewidth=4, label=f'Maximum Point: ({round(q, 5)}, {Pre_q})')
        else:
            plt.scatter(q, Pre_q, color='black', linewidth=4, label=f'Point: ({round(q, 5)}, {Pre_q})')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Bit Error Rate (BER)')
        plt.ylabel('Residual Error Probability (Pre)')
        #plt.title('Monte Carlo Simulation')
        plt.grid(True)
        y_value_horizontal_line = 2**-r
        # Add a horizontal line at 2^-r
        plt.axhline(y=y_value_horizontal_line, color='black', linestyle='--', linewidth=2, label=r'$P_{re}(0.5)=2^{-16}$')
        plt.plot(x_values_test, y_values_test, color='green',linewidth=5, label='LFSR Stochastic Automaton')
        plt.plot(x_values_is, y_values_is, color='orange', linewidth=3, label=f'Importance Sampling for q={round(q, 5)}')
        plt.plot(x_values_mc, y_values_mc, color='blue',linewidth=3, label='Monte Carlo Simulation')
        
        # print(type(is_xy))
        # print(is_xy)
        # is_y_values_upper = []
        # for i in x_values_mc:
        #     if(float(i)>=1e-3 and float(i)<= 0.5):
        #         is_y_values_upper.append(is_xy[float(i)])
        
        # Mean Squared Error (MSE) between Monte Carlo and Importance Sampling for values in [10^-3;0.5]
        # mse_upper_mc_is = np.mean((np.array(y_values_mc) - np.array(is_y_values_upper))**2)
        # Calculate Pearson correlation coefficient between Monte Carlo and Importance Sampling for values in [10^-3;0.5]
        # pearson_corr_upper_mc_is = np.corrcoef(y_values_mc, is_y_values_upper)[0, 1]
        
        # Create first legend
        legend1 = plt.legend(loc='lower right')
        # Plotting a dummy point for the second legend
        # plt.plot([], [], ' ', label="Extra Legend")
        # legend2 = plt.legend([f'MC-IS for BER in [10^-3;0.5], MSE: {mse_upper_mc_is}, Pearson: {pearson_corr_upper_mc_is}'], loc='lower right')
        # legend2 = plt.legend([f'MC-IS for BER in [10^-3;0.5], MSE: {mse_upper_mc_is}'], loc='lower right')
        # Add both legends to the plot
        # plt.gca().add_artist(legend1)
        
        # directory = "Results/CAN-FD3/prova1/Graphs"
        # directory = "Results/CAN-FD1/prova2/Graphs"
        # directory = "Results/CAN-FD/Graphs"
        directory = "Results/Test3/prova5/Graphs"
        # directory = "Results/Test8/prova2/Graphs"
        filename = f"MC_IS_ANLTCL.pdf"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, filename),bbox_inches='tight', pad_inches=0)
        plt.close()

# output_name1 = "Results/Test8/prova2/Results_2024_05_22_00_26.json"
output_name1 = "Results/Test3/prova5/Results_2024_05_14_23_41.json"
with open(output_name1) as config_file1:
    output1 = json.load(config_file1)
# input_name1 = "test8.json"
input_name1 = "test3.json"
# q1="0.5"; r=5
q1="0.0029973964323"; r=16

with open(input_name1) as config_file1:
    input1 = json.load(config_file1)

pre_dict1 = output1['pre_dict']
is_dict1 = output1['is_dict']
maxPre_x1 = float(output1['max_ber'])

is_xy1 = {}
is_xy1 = importance_sampling(input1, is_dict1, float(q1))
plot_mc_anltcl_is(pre_dict1, is_xy1, float(q1), maxPre_x1, input1, r)

# for key in is_dict1.keys():
#     q1 = float(key)
#     is_xy1 = {}
#     is_xy1 = importance_sampling(input1, is_dict1, q1)
#     plot_mc_anltcl_is(pre_dict1, is_xy1, q1, maxPre_x1, input1, r)