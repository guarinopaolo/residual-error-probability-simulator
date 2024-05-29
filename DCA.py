import itertools
import json
from typing import List
import numpy as np

'''
Returns XOR of 'a' and 'b' (both of same length)
'''
def xor(a, b):
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
def mod2div(dividend, divisor):
    # Number of bits to be XORed at a time.
    pick = len(divisor)
    # Slicing the dividend to appropriate
    # length for particular step
    tmp = dividend[0: pick]
    while pick < len(dividend):
        if tmp[0] == '1':
            # replace the dividend by the result
            # of XOR and pull 1 bit down
            tmp = xor(divisor, tmp) + dividend[pick]
        else: # If leftmost bit is '0'
            # If the leftmost bit of the dividend (or the
            # part used in each step) is 0, the step cannot
            # use the regular divisor; we need to use an
            # all-0s divisor.
            tmp = xor('0'*pick, tmp) + dividend[pick]
        # increment pick to move further
        pick += 1
    # For the last n bits, we have to carry it out
    # normally as increased value of pick will cause
    # Index Out of Bounds.
    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0'*pick, tmp)
    checkword = tmp
    return checkword

"""
    Implementation of the CRC calculated by the chosen base protocol at data-link layer.
    Default: algorithm described in ISO 1898-1:2015 for CAN-FD.
    :param data: portion of the frame to consider when calculating the CRC
    :param gen_poly: general polynomial defined by the protocol
    :return: True if the error is undetectable, False otherwise
"""
def compute_crc(data, gen_poly):    
    # Applying the formula: fcs(x) = (nd(x)*x^r)mod(g(x)), where r = deg(g(x)) = len(G)-1
    r = len(gen_poly) - 1
    # Appends r zeroes at end of data
    appended_data = data + '0'*(r)
    remainder = np.array(mod2div(appended_data, gen_poly)).astype(int)
    # Append remainder in the original data
    # T = data + remainder
    if(remainder.sum()==0):
        return True
    else:
        return False

'''
Given 'error_masks' a list of error masks having a fixed number of high bits,
returns how many error masks of such list are undetectable and
a list containing the undetectable error masks.
'''
def estimate(config: dict, error_masks: List[str]):
    original_sample = (np.frombuffer(buffer=config['original_sample'].encode('utf-8'), dtype='int8').copy()) - 48
    samples_undetected = 0
    generatorDL = config['datalink_generator']
    generatorNESTED  = config['generator']
    undetected_frames = []
    application_code_enabled = bool(config['nested_crc'])
    n = 0
    for error_mask in error_masks:
        error_mask = (np.frombuffer(buffer=error_mask.encode('utf-8'), dtype='int8').copy()) - 48
        # Focus on computing Pre on ND_nested injected with errors (ref. article assumption)
        if application_code_enabled == True and np.any(error_mask[
                  config['payload_code_portion_start_bit']:config['payload_code_portion_end_bit']
                  ] == 1):
            injected_sample = abs(original_sample - error_mask)
            # FCS is isolated from error mask, in order to make it independent from the protocol,
            # There could be a frame in which the FCS does not immediately follow the payload.
            injected_sample_datalink = injected_sample[
                                       config['datalink_crc_calc_start_bit']:
                                       config['datalink_crc_calc_end_bit']].copy()
            injected_sample_datalink = np.append(
                injected_sample_datalink,
                injected_sample[config['datalink_fcs_start_bit']:config['datalink_fcs_end_bit']].copy()
            )
            # generatorDL = (np.frombuffer(buffer=config['datalink_generator'].encode('utf-8'),dtype='int8').copy()) - 48
            # datalink_check = compute_crc(data=injected_sample_datalink,gen_poly=generator)
            # 1st CHECK: CRC at DATALINK LAYER
            # 1st UNDETECTED ERROR: The error has been injected in the original sample BUT the CRC has not detected it.
            injected_sample_datalink_str = ''.join([str(x) for x in injected_sample_datalink])
            if(compute_crc(data=injected_sample_datalink_str,gen_poly=generatorDL) == True):
            #if np.all(datalink_check == 0):
                injected_payload = injected_sample[config['payload_start_bit']:config['payload_end_bit']].copy()
                # generator = (np.frombuffer(buffer=config['generator'].encode('utf-8'), dtype='int8').copy()) - 48
                injected_sequence_for_nested_crc = injected_payload[config['nested_crc_start_bit']:config['nested_crc_end_bit']]
                # 2nd CHECK: (nested) CRC at APPLICATION LAYER
                # 2nd UNDETECTED ERROR: The error has been injected BUT the (nested) CRC has not detected it.
                injected_sequence_for_nested_crc_str = ''.join([str(x) for x in injected_sequence_for_nested_crc])
                if(compute_crc(injected_sequence_for_nested_crc_str, generatorNESTED) == True):
                    n = n + 1
                    error_mask_str = ''.join([str(x) for x in error_mask])
                    undetected_frames.append(error_mask_str)
                    samples_undetected += 1
        # For simulations in which the frame does not have a nested CRC, just one CRC at datalink level.
        elif application_code_enabled == False and (np.any(error_mask[config['datalink_crc_calc_start_bit']:
            config['datalink_crc_calc_end_bit']] == 1)):
            injected_sample = abs(original_sample - error_mask)
            injected_sample_datalink = injected_sample[
                                       config['datalink_crc_calc_start_bit']:config[
                                           'datalink_crc_calc_end_bit']].copy()
            injected_sample_datalink = np.append(
                injected_sample_datalink,
                injected_sample[
                config['datalink_fcs_start_bit']:config['datalink_fcs_end_bit']].copy()
            )
            injected_sample_datalink_str = ''.join([str(x) for x in injected_sample_datalink])
            if(compute_crc(data=injected_sample_datalink_str,gen_poly=generatorDL) == True):
                n += 1
                samples_undetected += 1
                error_mask_str = ''.join([str(x) for x in error_mask])
                undetected_frames.append(error_mask_str)
    return samples_undetected, undetected_frames
    
'''
Performs Monte Carlo simulation computing the residual error probability for values of
BER between 10^-3 and starting_ber (taken from .json file).
BER gets iteratively decremented from starting_ber by following the rule:
new_ber = actual_ber - actual_ber/4
as long as new_ber >= 10^3.
'''
def simulate(config: dict, highBitsWithErrorMasks: dict):
    data_points = {}
    A = {} # Dictionary s.t. key = number of high bits (starting from 1) and value = number of undetected error masks having key high bits.
    ber = float(config['starting_ber'])
    stay_in_loop = True
    highBitsWithUndetectedErrorMasks = {}

    for i in range(1, telegram_length+1):
        combinations = highBitsWithErrorMasks[i]
        A[i], undetected_error_masks = estimate(config, combinations)
        highBitsWithUndetectedErrorMasks[i] = undetected_error_masks

    while stay_in_loop:
        Pre = 0
        for i in range(1, telegram_length+1):    
            Pre += A[i] * (ber**i) * ((1-ber)**(telegram_length-i))
        data_points[format(ber, '.13f')] = Pre
        ber = ber - ber / 4
        if ber < 1e-10:
            stay_in_loop = False
    data_points_sorted = {k: data_points[k] for k in sorted(data_points)}
    return data_points_sorted, highBitsWithUndetectedErrorMasks

'''
Generate all combinations of n-bit strings with exactly k bits set to 1.
Returns a list containing each possible combination given a specific k.
'''
def generate_combinations(n, k):
    combinations = []
    positions = range(n)
    for bits in itertools.combinations(positions, k):
        bitstring = ['0'] * n
        for bit in bits:
            bitstring[bit] = '1'
        combinations.append(''.join(bitstring))
    return combinations

# conf_name = "test5.json"
conf_name = "test8.json"
with open(conf_name) as config_file:
    config = json.load(config_file)
# Length of the original bit string
telegram_length = config['telegram_length']
highBitsWithErrorMasks = {}

# Generate all combinations
for i in range(1,telegram_length+1):
    combinations = generate_combinations(telegram_length, i)
    highBitsWithErrorMasks[i] = combinations

data_points, highBitsWithUndetectedErrorMasks = simulate(config, highBitsWithErrorMasks)

# Save results to a .json file
json_array = {
    'high_bits_with_error_masks': highBitsWithErrorMasks,
    'high_bits_with_undetected_error_masks': highBitsWithUndetectedErrorMasks,
    'data_points': data_points
}
array_to_file = json.dumps(json_array, indent=4)
filename = "direct_code_analysis.json"
with open(filename, "w") as outfile:
    outfile.write(array_to_file)