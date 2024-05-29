import math

# sum = 0
# for i in range(1, 637):
#     num = math.factorial(636)
#     den = i*math.factorial(636-i)
#     C = num/den
#     sum += C
# 
# print(sum)

sum_value = 0
# n = 15 # 32767
# n = 20 # 1048575
# n = 30 # 1073741823
n = 40 # 1099511627775
# n = 50 # 1125899906842623
# n = 100 # 1267650600228229401496703205375
# n = 344 # 35835915874844867368919076489095108449946327955754392558399825615420669938882575126094039892345713852415
# n = 636 # 285152538601387201165073225356268207805826781703034995661199532368704697950542336656619550707335712486165144348349650456918044045085964874890791332482638386765749667147516559380179637015412735

# Start with the first binomial coefficient
C = 1  # This is \binom{636}{0}, and we use it to start the iteration

# Iterate to compute each \binom{636}{i} and add to the sum
for i in range(1, n + 1):
    C = C * (n - i + 1) // i  # Update to the next binomial coefficient
    sum_value += C

print(sum_value)
