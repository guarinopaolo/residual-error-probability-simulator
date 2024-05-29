# Residual Error Probability Simulator

Estimate the residual error probability with Monte Carlo simulations of the transmission of frames and extending the results with Importance Sampling.

1'000'000 iterations, 636 bits:
python3 main.py test3.json False Results/Test3/prova1
python3 main.py test3.json False Results/Test3/prova2
python3 main.py test3.json False Results/Test3/prova3
python3 main.py test3.json False Results/Test3/prova4
python3 main.py test3.json False Results/Test3/prova5
python3 main.py test3.json False Results/Test3/prova6
python3 main.py test3.json False Results/Test3/prova7

1'000'000 iterations, 7 bits, BERs taken from "test" dictionary:
python3 main.py test5.json False Results/Test5/prova1
python3 main.py test5.json False Results/Test5/prova2
python3 main.py test5.json False Results/Test5/prova3

1'000'000 iterations, 7 bits, BERs computed using ber = ber - ber/4:
python3 main.py test6.json False Results/Test6/prova1
python3 main.py test6.json False Results/Test6/prova2
python3 main.py test6.json False Results/Test6/prova3
python3 main.py test6.json False Results/Test6/prova4
python3 main.py test6.json False Results/Test6/prova5
python3 main.py test6.json False Results/Test6/prova6

10'000'000 iterations, 7 bits, BERs computed using ber = ber - ber/4:
python3 main.py test7.json False Results/Test7/prova1

100'000'000 iterations, 7 bits, BERs computed using ber = ber - ber/4:
python3 main.py test8.json False Results/Test8/prova1

1'000'000 iterations, 344 bits:
python3 main.py test1.json False Results/Test1/prova1

10'000'000 iterations, 344 bits:
python3 main.py test1.json False Results/Test1/prova2

1'000'000 iterations, 15 bits:
python3 main.py test8.json False Results/Test8/prova2

1'000'000 iterations, CAN-FD1, 148 bits bits:
python3 main.py CAN-FD1.json False Results/CAN-FD1/prova1

10'000'000 iterations, CAN-FD1, 148 bits bits:
python3 main.py CAN-FD1.json False Results/CAN-FD1/prova2