# Solving Differential Equations with Neural Networks

**Reproduction of the paper:** Lagaris, I., Likas, A., & Fotiadis, D. 1997, Computer Physics Communications,104, 1

**Instruction**: To see results check the NNDE notebook. To see the implementation of the network see ShallowNetwork.py.

**To-to-do list:**

* Consider other method of populating derivative tensors (numpy.from_function?)
* Refactor training methods of the Shallow Network class into a class Optimizer.
* Abstract the learning components to be provided for a task (parameters update rules, etc.).
