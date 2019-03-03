# Solving Differential Equations with Neural Networks

**Reproduction of the paper:** Lagaris, I., Likas, A., & Fotiadis, D. 1997, Computer Physics Communications,104, 1

**Current problem:** The solution seems to converge to an extremely suboptimal minimum.
**Hypothesis:** The used optimization algorithm (online (1 element batch) steepest gradient with momentum and decreasing learning ratio) is not suited for the task.
**Solution:** Use the method proposed by the authors - BFGS algorithm.

**Instruction**: To see results check the NNDE notebook. To see implementation of the network see ShallowNetwork.py.


**To-to-do list:**
* **Implement BFGS learning**.
* Conisider other method of populating derivative tensors (numpy.from_function?)
* Refactor training methods of the Shallow Network class into a class Optimizer.
* Abstract the learning components to be provided for a task (parameters update rules, etc.).
* Refactor the ShallowNetwork.py into many files.
