import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum

def importance_sampling(is_dict: dict, q: float):
    # starting_ber = float(self.config['starting_ber'])
    # starting_ber = 0.5
    # ber = starting_ber - starting_ber/4
    # probs = [starting_ber, ber]
    # while(ber>1e-10):
    #     ber = ber - ber/4
    #     if(ber >= 1e-10):
    #         probs.insert(0, ber)
    
    # conf_name = "CAN-FD3.json"
    conf_name = "CAN-FD1.json"
    with open(conf_name) as config_file:
            config = json.load(config_file)
    probs = []
    test = config['test']

    for key in test.keys():
        probs.append(float(key))
    probs.sort()

    estimator_sum = 0
    ListSumBitsErrorMask = is_dict[str(q)]
    is_values = {}
    for p in probs:
        for i in ListSumBitsErrorMask:
            # formula 5.1 and 5.2
            # estimator_sum += pow((p / q), i) * pow(((1 - p) / (1 - q)), (int(len(self.config['original_sample'])) - i))
            estimator_sum += pow((p / q), i) * pow(((1 - p) / (1 - q)), (int(len(config['original_sample'])) - i))
        # theta (5.2), estimation of Pre
        estimated_prob = float(estimator_sum) / float(config['iterations'])
        is_values[p] = estimated_prob

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
        directory = "Results/CAN-FD1/prova2/Graphs"
        filename = f"MC_and_IS_{q}.pdf"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, filename))
        plt.close()

output_name = "Results/CAN-FD1/prova2/Results_2024_05_24_00_00.json"
with open(output_name) as config_file:
    output = json.load(config_file)

pre_dict = output['pre_dict']
is_dict = output['is_dict']
maxPre_x = float(output['max_ber'])

is_xy = {}
for key in pre_dict.keys():
    is_xy = importance_sampling(is_dict, float(key))
    plot_mc_and_is(pre_dict, is_xy, float(key), maxPre_x)
