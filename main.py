import os
import sys
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt

from SimulationTools import ChannelSimulator, PlotVisualizer, NpEncoder, SimulationMode

if __name__ == "__main__":
    # python3 main.py test3.json False Results/Test3/prova1
    abs_conf_path = os.path.abspath(sys.argv[1])
    if sys.argv[2] != 'False':
        abs_reference_path = os.path.abspath(sys.argv[2])
    else:
        abs_reference_path = False
    Path(sys.argv[3]).mkdir(parents=True, exist_ok=True)
    os.chdir(sys.argv[3])   # Moves to Results/Test3/prova1
    time_start = datetime.now()

    chann_sim = ChannelSimulator(abs_conf_path)
    # pre_dict, is_dict, ber_with_n = chann_sim.simulate_test_values()
    # pre_dict, is_dict, ber_with_n = chann_sim.simulate_toy()
    pre_dict, is_dict, ber_with_n = chann_sim.simulate_test_values()

    # Find BER for which Pre is maximum.
    max_pre_x = chann_sim.search_maximum_pre(pre_dict)

    # Plot Monte Carlo results.
    pv = PlotVisualizer(abs_conf_path)
    pv.plot_montecarlo(pre_dict, max_pre_x)
    # pv.plot_mc_and_anlytcl(pre_dict, max_pre_x)
    
    # Plot Importance Sampling for a BER having at least two zeros after the floating point (i.e., 0.001234...).
    pv.plot_importance_sampling(pre_dict, is_dict, max_pre_x)

    # Apply Importance Sampling for each BER calculated and plot the resulting curve with the Monte Carlo reference
    # for key in is_dict.keys():
    #     pv.plot_importance_sampling_all_ber(results_array=is_dict[key], q=float(key), compare_results=pre_dict)
    # n, is_max = chann_sim.estimate(max_pre_x, SimulationMode.IMPORTANCE_SAMPLING)
    # pv.plot_importance_sampling_all_ber(results_array=is_max, q=float(max_pre_x), compare_results=pre_dict)
    
    # Estimate the Hamming Distance for the generator polynomial under analysis.
    chann_sim.estimate_hd()

    # Save output data structures in file json
    json_array = {
        'max_ber': max_pre_x, 
        'ber_with_undetected_masks': ber_with_n,
        'pre_dict': pre_dict, 
        'is_dict': is_dict
    }
    array_to_file = json.dumps(json_array, indent=4, cls=NpEncoder)
    filename = datetime.now().strftime("Results_%Y_%m_%d_%H_%M.json")
    with open(filename, "w") as outfile:
        outfile.write(array_to_file)
    print("The execution just finished. It took:", (datetime.now() - time_start))