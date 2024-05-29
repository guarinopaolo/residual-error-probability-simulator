import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from enum import Enum

# Returns XOR of 'a' and 'b'
# (both of same length)
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

# Performs Modulo-2 division
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

# Function used at the sender side to encode
# data by appending remainder of modular division
# at the end of data.
def encodeData(data, key):
	l_key = len(key)
	# Appends n-1 zeroes at end of data
	appended_data = data + '0'*(l_key-1)
	remainder = mod2div(appended_data, key)
	# Append remainder in the original data
	codeword = data + remainder
	print("Remainder : ", remainder)
	print("Encoded Data (Data + Remainder) : ",codeword)

"""
    Implementation of the CRC calculated by the chosen base protocol at data-link layer.
    Default: algorithm described in ISO 1898-1:2015 for CAN-FD.
    :param data: portion of the frame to consider when calculating the CRC
    :param gen_poly: general polynomial defined by the protocol
    :return: True if the error is undetectable, False otherwise
"""
def compute_crc(data, gen_poly):    
    # crc_reg = np.array([1]) # Single-element array: [1], representing the most significative bit to add to gen_poly
    # crc_reg = np.append(crc_reg, np.zeros(gen_poly.size - 1)) # crc_reg = [ 1 0 0 0 0 0 0 0 ]
    # for bit in data:
    #     crc_next = abs(bit - crc_reg[0])
    #     crc_reg = np.roll(crc_reg, -1)
    #     crc_reg[-1] = 0
    #     if crc_next == 1:
    #         reg_new = abs(crc_reg - gen_poly)
    #         crc_reg = np.copy(reg_new)
    # return crc_reg.copy()
	# Applying the formula: fcs(x) = (nd(x)*x^r)mod(g(x)), where r = deg(g(x)) = len(G)-1
	r = len(gen_poly) - 1
	# Appends r zeroes at end of data
	appended_data = data + '0'*(r)
	remainder = np.array(mod2div(appended_data, gen_poly)).astype(int)
	# Append remainder in the original data
	# T = data + remainder
	# print("Remainder : ", remainder)
	# print("Encoded Data (Data + Remainder) : ",T)
	if(remainder.sum()==0):
		return True
	else:
		return False

file_name = "NewResults/Test3/prova3/FramesWithErrorMasks_ber_0.1.json"
with open(file_name, 'r') as config_file:
    config = json.load(config_file)

injectedErrors = {}
injectedErrors = config['frames']
undetectedErrors = {}
count = 0
G = "100110011"
G_nested = "101000001"
sample = ""

for key in injectedErrors:
	sample = injectedErrors[key]
	if(compute_crc(sample, G)==True):
		if(compute_crc(sample[0:116],G_nested)==True):
			count +=1
			undetectedErrors[key] = sample

json_array = {'undetected_errors': undetectedErrors}
array_to_file = json.dumps(json_array, indent=4)
#filename = datetime.now().strftime("FramesWithErroMasks_%Y_%m_%d_%H_%M.json")
filename = f"NewResults/Test3/prova3/UndetectedErrors_ber_0.1.json"
with open(filename, "w") as outfile:
    outfile.write(array_to_file)
# print(len(injectedErrors))	
print(count)