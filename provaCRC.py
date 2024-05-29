import os
import sys
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_crc(data: np.ndarray, gen_poly: np.ndarray):
    """
    Implementation of the CRC calculated by the chosen base protocol at data-link layer.
    Default: algorithm described in ISO 1898-1:2015 for CAN-FD.

    :param data: portion of the frame to consider when calculating the CRC
    :param gen_poly: general polynomial defined by the protocol
    :return: frame check sequence
    """
    crc_reg = np.array([1]) # Single-element array: [1], representing the most significative bit to add to gen_poly
    crc_reg = np.append(crc_reg, np.zeros(gen_poly.size - 1)) # crc_reg = [ 1 0 0 0 0 0 0 0 ]
    for bit in data:
        crc_next = abs(bit - crc_reg[0])
        crc_reg = np.roll(crc_reg, -1)
        crc_reg[-1] = 0
        if crc_next == 1:
            reg_new = abs(crc_reg - gen_poly)
            crc_reg = np.copy(reg_new)
    return crc_reg.copy()

def check_received_payload(payload: np.ndarray):
    """
    Implementation of the code chosen for the study. The arguments should be modified accordingly.
    Default: call the ISO CAN FD algorithm.

    :param payload: portion of the frame to consider when calculating the CRC
    :return: True if the error is undetectable, False otherwise
    """
    # For test 1:
    # generator = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    # injected_sequence_for_nested_crc = payload[0:272]
    
    # For test 2:
    generator = np.array([0, 1, 0, 0, 0, 0, 0, 1])
    injected_sequence_for_nested_crc = payload[0:116]
    
    application_check = compute_crc(data=injected_sequence_for_nested_crc, gen_poly=generator)
    if np.all(application_check) == 0:
        return True     # Remainder = 0
    else:
        return False    # Remainder != 0

#                     NDstart                                                                                                     NDend|FCS     |                  
original_sample_str = "111101010100100101011000100010101111000110000111001111011111111100100111010011101101111110100110011001010001000010000100010110110011100111101110011101010001010001011001110110101010001100001011110001101110000001101101011000001010011010000010010000101110111011100011111110110011010100011001100001100001110101010011101101110110100100000101010001001100010101011101000111111110111111000111011110111011111111110111011001111001111011111000000001010010000111001111111111011001100101011101101101101111010001010011101000101110110010100011101000001000001010000010100100110000010010100000100111101001101001000001001000001101101001001100010000001111"
original_sample = (np.frombuffer(buffer=original_sample_str.encode('utf-8'), dtype='int8').copy()) - 48

# 1st PROOF for the CRC to be WRONG.
# error_sample_str = "000000000100100101011000100010101111000110000111001111011111111100100111010011101101111110100110011001010001000010000100010110110011100111101110011101010001010001011001110110101010001100001011110001101110000001101101011000001010011010000010010000101110111011100011111110110011010100011001100001100001110101010011101101110110100100000101010001001100010101011101000111111110111111000111011110111011111111110111011001111001111011111000000001010010000111001111111111011001100101011101101101101111010001010011101000101110110010100011101000001000001010000010100100110000010010100000100111101001101001000001001000001101101001001100010000001111"
# error_sample = (np.frombuffer(buffer=error_sample_str.encode('utf-8'), dtype='int8').copy()) - 48
# a = check_received_payload(error_sample)
# print(a)

# 2nd PROOF for the CRC to be WRONG.
ND_str = "100100"
G_str = "1101"
# FCS = (nd(x)*x^3)mod(g(x)) = x^0 = 001
ND = (np.frombuffer(buffer=ND_str.encode('utf-8'), dtype='int8').copy()) - 48
G = (np.frombuffer(buffer=G_str.encode('utf-8'), dtype='int8').copy()) - 48
print(compute_crc(ND, G))