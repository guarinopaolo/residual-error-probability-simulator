import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class SimulationMode(Enum):
    """
    FULL_MC:    get the results of a full Monte Carlo simulation
                (the counter of error masks and the sum of undetectable error bits)

    UNDETECTABLE_MASKS_COUNT:   get the number of undetectable error masks only

    IMPORTANCE_SAMPLING:    get the number of undetectable error masks and the results array needed to apply
                            the importance sampling technique
    """
    FULL_MC = 0
    UNDETECTABLE_MASKS_COUNT = 1
    IMPORTANCE_SAMPLING = 2

class ChannelSimulator:

    # def __init__(self, conf_name: str, application_code_enabled: bool):
    def __init__(self, conf_name: str):
        # self.application_code_enabled = application_code_enabled
        with open(conf_name) as config_file:
            self.config = json.load(config_file)
        self.application_code_enabled = bool(self.config['nested_crc'])

    '''
    Returns XOR of 'a' and 'b' (both of same length)
    '''
    def xor(self, a, b):
        # initialize result
        result = []
        # Traverse all bits, if bits are
        # same, then XOR is 0, else 1
        for i in range(1, len(b)):
            if a[i] == b[i]:
                result.append('0')
            else:
                result.append('1')
        return ''.join(result)

    '''
    Performs Modulo-2 division
    '''
    def mod2div(self, dividend, divisor):
        # Number of bits to be XORed at a time.
        pick = len(divisor)
        # Slicing the dividend to appropriate
        # length for particular step
        tmp = dividend[0: pick]
        while pick < len(dividend):
            if tmp[0] == '1':
                # replace the dividend by the result
                # of XOR and pull 1 bit down
                tmp = self.xor(divisor, tmp) + dividend[pick]
            else: # If leftmost bit is '0'
                # If the leftmost bit of the dividend (or the
                # part used in each step) is 0, the step cannot
                # use the regular divisor; we need to use an
                # all-0s divisor.
                tmp = self.xor('0'*pick, tmp) + dividend[pick]
            # increment pick to move further
            pick += 1
        # For the last n bits, we have to carry it out
        # normally as increased value of pick will cause
        # Index Out of Bounds.
        if tmp[0] == '1':
            tmp = self.xor(divisor, tmp)
        else:
            tmp = self.xor('0'*pick, tmp)
        checkword = tmp
        return checkword
    
    """
        Implementation of the CRC calculated by the chosen base protocol at data-link layer.
        Default: algorithm described in ISO 1898-1:2015 for CAN-FD.
        :param data: portion of the frame to consider when calculating the CRC
        :param gen_poly: general polynomial defined by the protocol
        :return: True if the error is undetectable, False otherwise
    """
    def compute_crc(self, data, gen_poly):    
        # Applying the formula: fcs(x) = (nd(x)*x^r)mod(g(x)), where r = deg(g(x)) = len(G)-1
        r = len(gen_poly) - 1
        # Appends r zeroes at end of data
        appended_data = data + '0'*(r)
        remainder = np.array(self.mod2div(appended_data, gen_poly)).astype(int)
        # Append remainder in the original data
        # T = data + remainder
        # print("Remainder : ", remainder)
        # print("Encoded Data (Data + Remainder) : ",T)
        if(remainder.sum()==0):
            return True
        else:
            return False

    '''
    Loads the orginal frame and generates #iterations error masks.
    3 simulation modes:
        1. FULL_MC, returns the results of a complete MC simulation
            -> #undetectableErrorMasks
            -> sum of undetectable error bits
        2. UNDETECTABLE_MASKS_COUNT, returns just
            -> #undetectableErrorMasks
        3. IMPORTANCE_SAMPLING, returns
            -> #undetectableErrorMasks
            -> array to apply the IS procedure
    '''
    def estimate(self, ber: float, simulation_mode: SimulationMode):
        #original_sample_str = self.config['original_sample']
        original_sample = (np.frombuffer(buffer=self.config['original_sample'].encode('utf-8'), dtype='int8').copy()) - 48
        #original_sample = self.config['original_sample']
        samples_undetected = 0
        sum_error_bits = 0
        results_array = []
        # generatorDL = (np.frombuffer(buffer=self.config['datalink_generator'].encode('utf-8'),dtype='int8').copy()) - 48
        # generatorNESTED  = (np.frombuffer(buffer=self.config['generator'].encode('utf-8'), dtype='int8').copy()) - 48
        generatorDL = self.config['datalink_generator']
        generatorNESTED  = self.config['generator']

        frames = {}
        undetected_frames = {}
        n = 0

        for i in range(self.config['iterations']):
            error_mask = np.random.binomial(1, ber, original_sample.size) # 636 for test 2
            # Focus on computing Pre on ND_nested injected with errors (ref. article assumption)
            if self.application_code_enabled == True and np.any(error_mask[
                      self.config['payload_code_portion_start_bit']:self.config['payload_code_portion_end_bit']
                      ] == 1):
                injected_sample = abs(original_sample - error_mask)
                
                error_mask_str = ''.join([str(x) for x in error_mask])
                injected_sample_str = ''.join([str(x) for x in injected_sample])
                frames[error_mask_str] = injected_sample_str
                
                # FCS is isolated from error mask, in order to make it independent from the protocol,
                # There could be a frame in which the FCS does not immediately follow the payload.
                injected_sample_datalink = injected_sample[
                                           self.config['datalink_crc_calc_start_bit']:
                                           self.config['datalink_crc_calc_end_bit']].copy()
                injected_sample_datalink = np.append(
                    injected_sample_datalink,
                    injected_sample[self.config['datalink_fcs_start_bit']:self.config['datalink_fcs_end_bit']].copy()
                )
                # generatorDL = (np.frombuffer(buffer=self.config['datalink_generator'].encode('utf-8'),dtype='int8').copy()) - 48
                # datalink_check = self.compute_crc(data=injected_sample_datalink,gen_poly=generator)
                # 1st CHECK: CRC at DATALINK LAYER
                # 1st UNDETECTED ERROR: The error has been injected in the original sample BUT the CRC has not detected it.
                injected_sample_datalink_str = ''.join([str(x) for x in injected_sample_datalink])
                if(self.compute_crc(data=injected_sample_datalink_str,gen_poly=generatorDL) == True):
                #if np.all(datalink_check == 0):
                    injected_payload = injected_sample[self.config['payload_start_bit']:self.config['payload_end_bit']].copy()
                    # generator = (np.frombuffer(buffer=self.config['generator'].encode('utf-8'), dtype='int8').copy()) - 48
                    injected_sequence_for_nested_crc = injected_payload[self.config['nested_crc_start_bit']:self.config['nested_crc_end_bit']]
                    # 2nd CHECK: (nested) CRC at APPLICATION LAYER
                    # 2nd UNDETECTED ERROR: The error has been injected BUT the (nested) CRC has not detected it.
                    injected_sequence_for_nested_crc_str = ''.join([str(x) for x in injected_sequence_for_nested_crc])
                    if(self.compute_crc(injected_sequence_for_nested_crc_str, generatorNESTED) == True):
                        n = n + 1
                        undetected_frames[error_mask_str] = injected_sample_str
                        if simulation_mode == SimulationMode.FULL_MC:
                            samples_undetected += 1
                            sum_error_bits += (error_mask[self.config['datalink_crc_calc_start_bit']:self.config[
                                'datalink_crc_calc_end_bit']].sum() + error_mask[
                                                                      self.config['datalink_fcs_start_bit']:self.config[
                                                                          'datalink_fcs_end_bit']].sum())
                        elif simulation_mode == SimulationMode.UNDETECTABLE_MASKS_COUNT:
                            samples_undetected += 1
                        elif simulation_mode == SimulationMode.IMPORTANCE_SAMPLING:
                            samples_undetected += 1
                            results_array.append(error_mask.sum())
            # For simulations in which the frame does not have a nested CRC, just one CRC at datalink level.
            elif self.application_code_enabled == False and (np.any(error_mask[self.config['datalink_crc_calc_start_bit']:
                self.config['datalink_crc_calc_end_bit']] == 1)):
                injected_sample = abs(original_sample - error_mask)
                error_mask_str = ''.join([str(x) for x in error_mask])
                injected_sample_str = ''.join([str(x) for x in injected_sample])
                frames[error_mask_str] = injected_sample_str
                injected_sample_datalink = injected_sample[
                                           self.config['datalink_crc_calc_start_bit']:self.config[
                                               'datalink_crc_calc_end_bit']
                                           ].copy()
                injected_sample_datalink = np.append(
                    injected_sample_datalink,
                    injected_sample[
                    self.config['datalink_fcs_start_bit']:self.config['datalink_fcs_end_bit']
                    ].copy()
                )
                # generatorDL = (np.frombuffer(buffer=self.config['datalink_generator'].encode('utf-8'),dtype='int8').copy()) - 48
                # datalink_check = self.compute_crc(data=injected_sample_datalink,gen_poly=generator)
                injected_sample_datalink_str = ''.join([str(x) for x in injected_sample_datalink])
                if(self.compute_crc(data=injected_sample_datalink_str,gen_poly=generatorDL) == True):
                    n = n + 1
                    undetected_frames[error_mask_str] = injected_sample_str
                    if simulation_mode == SimulationMode.FULL_MC:
                        samples_undetected += 1
                        # In order to parametrize the sum of the bits of the error masks for frames having the FCS sequence
                        # not following the payload.
                        sum_error_bits += (error_mask[self.config['datalink_crc_calc_start_bit']:self.config[
                            'datalink_crc_calc_end_bit']].sum() + error_mask[
                                                                  self.config['datalink_fcs_start_bit']:self.config[
                                                                      'datalink_fcs_end_bit']].sum())
                        # sum_error_bits += error_mask.sum() isn't the same?
                    elif simulation_mode == SimulationMode.UNDETECTABLE_MASKS_COUNT:
                        samples_undetected += 1
                    elif simulation_mode == SimulationMode.IMPORTANCE_SAMPLING:
                        samples_undetected += 1
                        results_array.append(error_mask.sum())

        print('ber: '+ str(ber) + '\t\t\tn: ' + str(n))
        # Print to file the generated error masks per BER. Each line will be in the form error_mask:injected_frame.
        # json_array = {'frames': frames}
        # array_to_file = json.dumps(json_array, indent=4, cls=NpEncoder)
        # directory = "ErrorMasks"
        # filename = f"FramesWithErrorMasks_ber_{round(ber, 3)}.json"
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # with open(os.path.join(directory, filename), 'w') as file:
        #     file.write(array_to_file)

        # Print to file the undetected error masks generated per BER. Each line will be in the form error_mask:injected_frame.
        if n > 0:
            json_array = {'undetected_frames': undetected_frames}
            array_to_file = json.dumps(json_array, indent=4, cls=NpEncoder)
            directory = "UndetectedErrorMasks"
            filename = f"UndetectedEMs_ber_{round(ber, 3)}.json"
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(os.path.join(directory, filename), 'w') as file:
                file.write(array_to_file)

        if simulation_mode == SimulationMode.FULL_MC:
            return samples_undetected, sum_error_bits
        elif simulation_mode == SimulationMode.UNDETECTABLE_MASKS_COUNT:
            return samples_undetected
        elif simulation_mode == SimulationMode.IMPORTANCE_SAMPLING:
            return samples_undetected, results_array

    '''
    Performs Monte Carlo simulation computing the residual error probability for values of
    BER between 10^-3 and starting_ber (taken from .json file).
    BER gets iteratively decremented from starting_ber by following the rule:
    new_ber = actual_ber - actual_ber/4
    as long as new_ber >= 10^3.
    '''
    def simulate(self):
        # data_points_mc = [] # list of tuples
        time_start = datetime.now()
        data_points_mc = {}
        is_dict = {}
        ber_with_n = {}
        Pre = 0
        ListBitSumPerErrorMask = []
        tot_iter = self.config['iterations']
        ber = self.config['starting_ber']
        stay_in_loop = True

        while stay_in_loop:
            samples_undetected, ListBitSumPerErrorMask = self.estimate(ber, SimulationMode.IMPORTANCE_SAMPLING)
            Pre = samples_undetected/tot_iter
            ber_with_n[ber]=samples_undetected
            # data_points_mc[format(ber, '.13f')] = Pre
            # is_dict[format(ber, '.13f')] = ListBitSumPerErrorMask
            data_points_mc[str(ber)] = Pre
            is_dict[str(ber)] = ListBitSumPerErrorMask
            ber = ber - ber / 4
            if ber < 1e-3:
                stay_in_loop = False
        print("The execution of simulate() took:", (datetime.now() - time_start))
        data_points_mc_sorted = {k: data_points_mc[k] for k in sorted(data_points_mc)}
        is_dict_sorted = {k: is_dict[k] for k in sorted(is_dict)}
        return data_points_mc_sorted, is_dict_sorted, ber_with_n
    
    '''
    Performs Monte Carlo simulation computing the residual error probability for values of
    BER between 10^-3 and 0.5 taken from 'test' dictionary inside the input .json file.
    '''
    def simulate_test_values(self):
        time_start = datetime.now()
        data_points_mc = {}
        is_dict = {}
        ber_with_n = {}
        Pre = 0
        ListBitSumPerErrorMask = []
        tot_iter = self.config['iterations']
        ber = self.config['starting_ber']
        sub_test = {}
        test = self.config['test']
        for key in test:
            ber = float(key)
            if(ber>= 1e-3 and ber <= 0.5):
                sub_test[ber] = float(test[key])
        
        for key in sub_test:
            samples_undetected, ListBitSumPerErrorMask = self.estimate(key, SimulationMode.IMPORTANCE_SAMPLING)
            Pre = samples_undetected/tot_iter
            ber_with_n[key]=samples_undetected
            # data_points_mc[format(key, '.13f')] = Pre
            # is_dict[format(key, '.13f')] = ListBitSumPerErrorMask
            data_points_mc[str(key)] = Pre
            is_dict[str(key)] = ListBitSumPerErrorMask


        print("The execution of simulate() took:", (datetime.now() - time_start))
        data_points_mc_sorted = {k: data_points_mc[k] for k in sorted(data_points_mc)}
        is_dict_sorted = {k: is_dict[k] for k in sorted(is_dict)}
        return data_points_mc_sorted, is_dict_sorted, ber_with_n

    '''
    Finds the value of BER for which the residual error probability (Pre) is maximum.
    '''
    def search_maximum_pre(self, pre_dict: dict):
        # Find maximum point
        list_data_points_mc = list(pre_dict.items())
        # The function lambda point: point[1] extracts the y-value from each tuple.
        # Inside the lambda function, point represents each tuple in the list, and point[1] accesses the y-value.
        # So, the lambda function effectively transforms each tuple into its y-value.
        max_point = max(list_data_points_mc, key=lambda point: point[1])
        max_ber = float(max_point[0])
        return max_ber
    
    '''
    Computes the estimation of the Hamming distance by summing up the number of high bits
    of the undetected error mask having the lower number of high bits among all the
    undetected error masks.
    The output is printed in a file named: HD.json
    '''
    def estimate_hd(self):
        directory = "UndetectedErrorMasks"
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
        filename = "HD.json"
        with open(filename, "w") as outfile:
            outfile.write(array_to_file)

class PlotVisualizer:
    def __init__(self, conf_name: str):
        with open(conf_name) as config_file:
            self.config = json.load(config_file)

    def plot_montecarlo(self, pre_dict: dict, maxPre_x: float):
        x_values_mc = list(pre_dict.keys())
        x_values_mc = [float(element) for element in x_values_mc]
        y_values_mc = list(pre_dict.values())
        y_values_mc = [float(element) for element in y_values_mc]
        maxPre_y = float(y_values_mc[x_values_mc.index(maxPre_x)])
        bounded_x_values = [num for num in x_values_mc if '0.00' in str(num)]
        bounded_y_values = []
        for i in bounded_x_values:
            bounded_y_values.append(y_values_mc[x_values_mc.index(i)])
        # Remove the points having 0 as y-coordinate (i.e., Pre=0)
        # Sort points based on x-coordinate
        bounded_values = list(zip(bounded_x_values, bounded_y_values))
        # print(bounded_x_values)
        # print(bounded_y_values)
        bounded_values = [(x_val, y_val) for x_val, y_val in zip(bounded_x_values, bounded_y_values) if y_val != 0]
        bounded_x_values = [point[0] for point in bounded_values]
        bounded_y_values = [point[1] for point in bounded_values]
        # print(bounded_x_values)
        # print(bounded_y_values)
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
        directory = "Graphs"
        filename = 'MC.pdf'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, filename))

    '''
    Procedure that plots both Monte Carlo and Analytical values.
    To be invoked only if 'simulate_test_values()' has been used to perform
    the Monte Carlo simulation.
    '''
    def plot_mc_and_anlytcl(self, pre_dict: dict, maxPre_x: float):
        x_values_mc = list(pre_dict.keys())
        x_values_mc = [float(element) for element in x_values_mc]
        y_values_mc = list(pre_dict.values())
        y_values_mc = [float(element) for element in y_values_mc]

        test = self.config['test']
        sub_test = {}
        for key in test:
            ber = float(key)
            if(ber>= 1e-3 and ber <= 0.5):
                sub_test[ber] = float(test[key])
        sorted_x = sorted(sub_test.keys())
        sorted_sub_test = {key: sub_test[key] for key in sorted_x}
        x_values = list(sorted_sub_test.keys())
        y_values = list(sorted_sub_test.values())

        # Mean Squared Error (MSE) between the analytical curve and Monte Carlo results
        mse = np.mean((np.array(y_values) - np.array(y_values_mc))**2)
        # Calculate Pearson correlation coefficient between the analytical curve and Monte Carlo results
        pearson_corr = np.corrcoef(y_values, y_values_mc)[0, 1]

        maxPre_y = float(y_values_mc[x_values_mc.index(maxPre_x)])
        bounded_x_values = [num for num in x_values_mc if '0.00' in str(num)]
        bounded_y_values = []
        for i in bounded_x_values:
            bounded_y_values.append(y_values_mc[x_values_mc.index(i)])
        # Remove the points having 0 as y-coordinate (i.e., Pre=0)
        # Sort points based on x-coordinate
        bounded_values = list(zip(bounded_x_values, bounded_y_values))
        # print(bounded_x_values)
        # print(bounded_y_values)
        bounded_values = [(x_val, y_val) for x_val, y_val in zip(bounded_x_values, bounded_y_values) if y_val != 0]
        bounded_x_values = [point[0] for point in bounded_values]
        bounded_y_values = [point[1] for point in bounded_values]
        # print(bounded_x_values)
        # print(bounded_y_values)
        plt.figure(figsize=(15, 8))
        for x, y in zip(bounded_x_values, bounded_y_values):
            plt.scatter(x, y, color='green', label=f'Point: ({round(x, 5)}, {y})')
        plt.yscale('log')
        plt.xscale('log')
        plt.scatter(maxPre_x, maxPre_y, color='red', label=f'Maximum Point: ({round(maxPre_x,5)}, {maxPre_y})')
        plt.xlabel('Bit Error Rate (BER)')
        plt.ylabel('Residual Error Probability (Pre)')
        #plt.title('Monte Carlo Simulation')
        plt.grid(True)
        plt.plot(x_values, y_values, color='green', label='Analytical Curve')
        plt.plot(x_values_mc, y_values_mc, color='blue', label='Monte Carlo Simulation')
        
        # legend_label = f'MSE: {mse:.2f}, Pearson: {pearson_corr:.2f}'
        # plt.legend([legend_label])
        # plt.legend()
        
        # Create first legend
        legend1 = plt.legend(loc='upper left')
        # Plotting a dummy point for the second legend
        plt.plot([], [], ' ', label="Extra Legend")
        legend2 = plt.legend([f'MSE: {mse}, Pearson: {pearson_corr}'], loc='lower right')
        # Add both legends to the plot
        plt.gca().add_artist(legend1)
        
        #plt.show()

        directory = "Graphs"
        filename = 'MC_and_ANLTCL.pdf'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, filename))
        # filename = 'MC.pdf'
        # plt.savefig(filename)
    
    def plotIS(self, y_values_mc: list, is_dict: dict, q: float):
        is_dict = {float(key): value for key, value in is_dict.items()}
        estimator_sum = 0
        # Projected probabilities from 0.5 down to 10^-10
        # probs = np.array([
        #     0.0000000001, 0.0000000005, 0.000000001, 0.000000005, 0.00000001, 0.00000005, 0.0000001, 0.0000005,
        #     0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5
        # ])
        test = self.config['test']
        swap = {}
        for key in test:
            ber = float(key)
            swap[ber] = float(test[key])
        sorted_x = sorted(swap.keys())
        test = {key: swap[key] for key in sorted_x}
        probs = np.array([float(key) for key in test.keys()]) 
        probs.sort() 
        is_values = {}
        ListSumBitsErrorMask = is_dict[q]
        for p in probs:
            estimator_sum = 0
            for item in ListSumBitsErrorMask:
                # formula 5.1 and 5.2
                estimator_sum += pow((p / q), item) * pow(((1 - p) / (1 - q)), (int(len(self.config['original_sample'])) - item))
            # theta (5.2), estimation of Pre
            estimated_prob = float(estimator_sum) / float(self.config['iterations'])
            is_values[p] = estimated_prob

        # Fixed formula for q = 0.5
        # for p in probs:
        #     estimator_sum = 0
        #     for item in ListSumBitsErrorMask:
        #         # formula 5.1
        #         p = float(p)
        #         item = int(item)
        #         estimator_sum += pow((p / (1-p)), item) * pow((2*(1-p)), len(self.config['original_sample']))
        #     # theta (5.2), estimation of Pre
        #     estimated_prob = float(estimator_sum) / float(self.config['iterations'])
        #     is_values[p] = estimated_prob
        
        # y_values_upper = []
        # for key in test.keys():
        #     if(float(key)>=1e-3):
        #         y_values_upper.append(float(test[key]))
        is_y_values_upper = []
        for key in is_values.keys():
            if(float(key)>=1e-3):
                is_y_values_upper.append(float(is_values[key]))
        # Mean Squared Error (MSE) between the analytical curve and Importance Sampling results for values in [10^-3;0.5]
        # mse_upper = np.mean((np.array(y_values_upper) - np.array(is_y_values_upper))**2)
        # Calculate Pearson correlation coefficient between the analytical curve and Importance Sampling results for values in [10^-3;0.5]
        # pearson_corr_upper = np.corrcoef(y_values_upper, is_y_values_upper)[0, 1]

        # Mean Squared Error (MSE) between Monte Carlo and Importance Sampling for values in [10^-3;0.5]
        mse_upper_mc_is = np.mean((np.array(y_values_mc) - np.array(is_y_values_upper))**2)
        # Calculate Pearson correlation coefficient between Monte Carlo and Importance Sampling for values in [10^-3;0.5]
        pearson_corr_upper_mc_is = np.corrcoef(y_values_mc, is_y_values_upper)[0, 1]

        # y_values_lower = []
        # for key in test.keys():
        #     if(float(key)<1e-3):
        #         y_values_lower.append(float(test[key]))
        # is_y_values_lower = []
        # for key in is_values.keys():
        #     if(float(key)<1e-3):
        #         is_y_values_lower.append(float(is_values[key]))
        # Mean Squared Error (MSE) between the analytical curve and Importance Sampling results for values in [10^-10;10^-3)
        # mse_lower = np.mean((np.array(y_values_lower) - np.array(is_y_values_lower))**2)
        # Calculate Pearson correlation coefficient between the analytical curve and Importance Sampling results for values in [10^-10;10^-3)
        # pearson_corr_lower = np.corrcoef(y_values_lower, is_y_values_lower)[0, 1]

        # Create first legend
        # legend1 = plt.legend([f'IS-Analytical for BER in [10^-3;0.5], MSE: {mse_upper}, Pearson: {pearson_corr_upper}\nIS-Analytical for BER in [10^-10;10^-3), MSE: {mse_lower}, Pearson: {pearson_corr_lower}\nMC-IS for BER in [10^-3;0.5], MSE: {mse_upper_mc_is}, Pearson: {pearson_corr_upper_mc_is}'], loc='lower right')
        legend1 = plt.legend([f'MC-IS for BER in [10^-3;0.5], MSE: {mse_upper_mc_is}, Pearson: {pearson_corr_upper_mc_is}'], loc='lower right')
        # Plotting a dummy point for the second legend
        # plt.plot([], [], ' ', label="Extra Legend")
        is_x_values = [float(x) for x in is_values.keys()]
        is_y_values = list(is_values.values())
        plt.plot(is_x_values, is_y_values, linewidth=2.5, c="orange", zorder=1, label='IS with q={}'.format(round(q,5)))
        return legend1

    # def plot(self, maxPre_x: float, maxPre_y: float, x_values_mc: list, y_values_mc: list,
    #           analytical_x_values: list, analytical_y_values: list, q: float, is_dict: dict):
    def plot(self, maxPre_x: float, maxPre_y: float, x_values_mc: list, y_values_mc: list, q: float, is_dict: dict):
        plt.yscale('log')
        plt.xscale('log')
        plt.scatter(maxPre_x, maxPre_y, color='red', label=f'Point of Maximum Pre (MC): ({round(maxPre_x,5)}, {maxPre_y})')
        plt.xlabel('Bit Error Rate (BER)')
        plt.ylabel('Residual Error Probability (Pre)')
        plt.title('Monte Carlo Simulation')
        plt.grid(True)
        plt.plot(x_values_mc, y_values_mc, color='blue', label='Monte Carlo Simulation')
        # plt.plot(analytical_x_values, analytical_y_values, color='green', label='Analytical Values')
        legend1 = self.plotIS(y_values_mc, is_dict, q)
        # plt.legend()

        legend2 = plt.legend(loc='upper left')
        # Add both legends to the plot
        # plt.gca().add_artist(legend1)

        directory = "Graphs"
        filename = 'IS_q_{}.pdf'.format(q)
        if not os.path.exists(directory):
            os.makedirs(directory)
        #filename = 'Graphs/IS_q_{}.pdf'.format(q)
        #plt.savefig(filename)
        plt.savefig(os.path.join(directory, filename))
        #plt.show()

    def plot_importance_sampling(self, pre_dict: dict, is_dict: dict, maxPre_x: float):
        print("Computing plot_imp_sampl()")
        x_values_mc = list(pre_dict.keys())
        x_values_mc = [float(element) for element in x_values_mc]
        y_values_mc = list(pre_dict.values())
        y_values_mc = [float(element) for element in y_values_mc]
        maxPre_y = float(y_values_mc[x_values_mc.index(maxPre_x)])
        # analytical_values = self.config['test']
        # analytical_x_values = [float(x) for x in analytical_values.keys()]
        # analytical_y_values = list(analytical_values.values())

        # bounded_x_values = [num for num in x_values_mc if '0.00' in str(num)]
        bounded_x_values = [num for num in x_values_mc if '0.' in str(num)]
        bounded_y_values = []
        for i in bounded_x_values:
            bounded_y_values.append(y_values_mc[x_values_mc.index(i)])
        # Remove the points having 0 as y-coordinate (i.e., Pre=0)
        # Sort points based on x-coordinate
        bounded_values = list(zip(bounded_x_values, bounded_y_values))
        # print(bounded_x_values)
        # print(bounded_y_values)
        bounded_values = [(x_val, y_val) for x_val, y_val in zip(bounded_x_values, bounded_y_values) if y_val != 0]
        bounded_x_values = [point[0] for point in bounded_values]
        bounded_y_values = [point[1] for point in bounded_values]
        # print(bounded_x_values)
        # print(bounded_y_values)
        for q in bounded_x_values:
            plt.figure(figsize=(15, 8))
            for x, y in zip(bounded_x_values, bounded_y_values):
                if(x == q):
                    plt.scatter(x, y, color='green', label=f'Point: ({round(x, 5)}, {y})')
            # print(float(q))
            # self.plot(maxPre_x, maxPre_y, x_values_mc, y_values_mc, 
            #           analytical_x_values, analytical_y_values, float(q), is_dict)
            if(q == bounded_x_values[len(bounded_values)-1]):
                plt.figure(figsize=(15, 8))
                # self.plot(maxPre_x, maxPre_y, x_values_mc, y_values_mc, 
                #       analytical_x_values, analytical_y_values, maxPre_x, is_dict)
                self.plot(maxPre_x, maxPre_y, x_values_mc, y_values_mc, maxPre_x, is_dict)

    def plot_importance_sampling_all_ber(self, results_array: np.ndarray, q: float, compare_results: dict):
        """
        results_array contains for each element the summation of the high bits for each errorMask which produces an undetected error.
        Hence, results_array contains |errorMask| elements.
        """
        estimator_sum = 0
        # Projected probabilities from 0.5 down to 10^-10
        probs = np.array([
            0.0000000001, 0.0000000005, 0.000000001, 0.000000005, 0.00000001, 0.00000005, 0.0000001, 0.0000005,
            0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5
        ])
        res = {}
        for p in probs:
            estimator_sum = 0
            for item in results_array:
                # formula 5.1 and 5.2
                estimator_sum += pow((p / q), item) * pow(((1 - p) / (1 - q)), (int(len(self.config['original_sample'])) - item))
            # theta (5.2), estimation of Pre
            estimated_prob = float(estimator_sum) / float(self.config['iterations'])
            res[p] = estimated_prob

        x_axis = np.array([])
        y_axis = np.array([])

        for key in res.keys():
            x_axis = np.append(x_axis, float(key))
            y_axis = np.append(y_axis, float(res[key]))
        fig, ax = plt.subplots()
        ax.plot(x_axis, y_axis, c="Blue", zorder=1, label='IS with q={}'.format(q))

        x_axis2 = np.array([])
        y_axis2 = np.array([])

        # Compare the two curves: Monte Carlo's one (from 0.5 to 10^-3) and Importance Sampl. (from 0.5 to 10^-10)
        for key in compare_results.keys():
            x_axis2 = np.append(x_axis2, float(key))
            y_axis2 = np.append(y_axis2, float(compare_results[key]) / float(self.config['iterations']))
        ax.plot(x_axis2, y_axis2, c="Magenta", zorder=2, label='MC')

        x_axis3 = np.array([])
        y_axis3 = np.array([])
        test = self.config['test']

        for key in test.keys():
            x_axis3 = np.append(x_axis3, float(key))
            y_axis3 = np.append(y_axis3, float(test[key]))
        ax.plot(x_axis3, y_axis3, c="Green", zorder=3, label='Analytical')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        filename = "Importance_Sampling_q_{}.pdf".format(q)
        plt.savefig(filename)
        plt.close(fig)
