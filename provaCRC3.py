import numpy as np

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

# PROOF of CORRECTNESS:
# data = "100100"
# data_injected = "100101"
# key = "1101"
# # encodeData(data, key)
# # encodeData(data_injected, key)
# T = "100100001"
# G = "1101"
# # encodeData(T,G)

# ND = "1110011"
# G = "1011"
# T = "1110011010"
# # encodeData(T, G)

# ND = "100100"
# G = "1101"
# encodeData(ND, G)
# # FCS = (nd(x)*x^3)mod(g(x)) = x^0 = 001
# T = ND + "001"
# encodeData(T, G)

# ND = "1101000100"
# G = "100101"
# encodeData(ND, G)
# # FCS = (nd(x)*x^3)mod(g(x)) = x^0 = 001
# T = ND + "10010"
# encodeData(T, G)

ND = "0110000000110010111001101110101101011110101111010100000010101001001000000011100101101001111100101110001111111010000001"
G = "110110100001011011"
print(f"LEN ND: {len(ND)}\nLEN G: {len(G)}")
# encodeData(ND, G)
FCS = "00101011101101101"
# FCS = (nd(x)*x^3)mod(g(x)) = x^0 = 001
T = ND + FCS
print(f"T: {T}\nLEN T: {len(T)}")
encodeData(T, G)

# TEST 1:
# Tp = "00101101100101101011100101100110000000001100101010010010001100111111100110110000101001010101100110100100100000110100111101111001101001100011101111100000110110001100000011111001000110010100000110000000111111110010111010011100010100000111100010110000011011010011001101001011011111100001001001110110111111111111101010100011111000011110110101001100"
# ND_nested = Tp[0:256]
# G_nested = "10100111010101011"
# # print(len(G_nested))
# FCS_nested_p = Tp[256:272] # "0011001101001011"
# # print('FCS_nested_p len: ', len(FCS_nested_p))
# # print('FCS_nested_p: ', FCS_nested_p)
# encodeData(ND_nested, G_nested)
# FCS_nested = "1010111000011101"
# T_nested = ND_nested + FCS_nested
# print(compute_crc(T_nested, G_nested))
# 
# ND = ND_nested + FCS_nested + Tp[272:336]
# print("ND len: ", len(ND))
# FCSp = Tp[336:344] # "01001100"
# # print("FCSp: ", FCSp)
# G = "111111111"
# encodeData(ND, G)
# FCS = "10101001"
# T = ND + FCS
# print(compute_crc(T,G))
# print("T: ", T)
# print(len(T))
# print(len(Tp))
# 
# payload = "00101101100101101011100101100110000000001100101010010010001100111111100110110000101001010101100110100100100000110100111101111001101001100011101111100000110110001100000011111001000110010100000110000000111111110010111010011100010100000111100010110000011011011010111000011101011111100001001001110110111111111111101010100011111000011110110110101001"
# print('payload len: ', len(payload))
# print('payload: ', payload)
# print(compute_crc(payload, G))

# TEST 2, 3, 4:
# Tp = "111101010100100101011000100010101111000110000111001111011111111100100111010011101101111110100110011001010001000010000100010110110011100111101110011101010001010001011001110110101010001100001011110001101110000001101101011000001010011010000010010000101110111011100011111110110011010100011001100001100001110101010011101101110110100100000101010001001100010101011101000111111110111111000111011110111011111111110111011001111001111011111000000001010010000111001111111111011001100101011101101101101111010001010011101000101110110010100011101000001000001010000010100100110000010010100000100111101001101001000001001000001101101001001100010000001111"
# ND_nested = Tp[0:108]
# G_nested = "101000001"
# FCS_nested_p = Tp[108:116]
# # print("FCS_nested_p: ", FCS_nested_p)
# # print(len(FCS_nested_p))
# # print(len(ND_nested))
# encodeData(ND_nested, G_nested)
# FCS_nested = "00000010"
# T_nested = ND_nested + FCS_nested
# # print(compute_crc(T_nested, G_nested))
# 
# G = "100110011"
# ND = ND_nested + FCS_nested + Tp[116:628]
# FCSp = Tp[628:636]
# print("FCSp: ", FCSp)
# print(len(FCSp))
# # print('Tp: ', Tp)
# # print('FCSp: ', FCSp)
# encodeData(ND, G)
# FCS = "10101010"
# T = ND + FCS
# print(compute_crc(T, G),"\n")
# print("T: ", T)
# print(len(T))
# 
# payload = "111101010100100101011000100010101111000110000111001111011111111100100111010011101101111110100110011001010001000000100100010110110011100111101110011101010001010001011001110110101010001100001011110001101110000001101101011000001010011010000010010000101110111011100011111110110011010100011001100001100001110101010011101101110110100100000101010001001100010101011101000111111110111111000111011110111011111111110111011001111001111011111000000001010010000111001111111111011001100101011101101101101111010001010011101000101110110010100011101000001000001010000010100100110000010010100000100111101001101001000001001000001101101001001100010010101010"
# print("payload: ", payload)
# print(len(payload))
# print(compute_crc(payload,G))