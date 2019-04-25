# Solving Differential Equations with Neural Networks

### **Reproduction of the papers:** 

[1] Lagaris, I., Likas, A., & Fotiadis, D. 1997, Computer Physics Communications,104, 1

## **Roadmap**: 
The repository contains three parts:
* Tensorflow version - code based on the Tensorflow library, enables simple :
    * Works:
        * ODEs with easy Dirichlet and Neumann initial conditions (Examples no. 1-3),
        * Systems of coupled ODEs (Example no. 4),
        * PDEs with Dirichlet boundary condition (Example no. 5),
    * To be done:
        * Neumann boundary conditions for PDEs (Examples no. 6-7 - written, not converging),
        * Custom weights initialization,
        * Better optimization procedure (BFGS),
        
* Numpy_version - code from scratch based only on numpy for solving differential equations via the trial solution with a shallow network:
    * Works:
        * ODEs with Dirichlet and Neumann boundary conditions Examples (no. 1-3),
        * PDEs with Dirichlet Conditions (Example no. 5),
    * To be done:
        * Neumann boundary conditions for PDEs,
        * Systems of coupled ODEs,
        * optimization of the tensor operations,

* Others - some other experiments:
    * Keras version - very inelastic and inelegant code.
