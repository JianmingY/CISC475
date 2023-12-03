To run the code, use the command line:

example: python .\train_1.py -op 1 -n 512 -lr 0.00005 -b 64 -e 50 -s 1 -r 1e-15 -d 0.3

-op is optimizer (from 1 to 3, represents Adam, SGD and RMSprop)
-n is number of nodes
-lr is learning rate
-b is batch size
-e is number of epochs
-s is scheduler (from 1 to 3: reduce on plateau, StepLR and OneCycle)
-r is L2 weight decay
-d is dropout rate
# CISC475
