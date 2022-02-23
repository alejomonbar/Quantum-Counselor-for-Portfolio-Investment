# Introduction


# Quantum Counselor for Portfolio Investment
The Quantum Counselor for portfolio investment is a tool with two main objectives: forecasting the trend of assets price and optimizing portfolio returns both using quantum computing techniques. For the case of the forecasting method, we use a hybrid method that combines a deep learning model of classical LSTM layers with quantum layers. For the case of portfolio optimization, the quantum algorithms of QAOA and VQE are used to solve the problem and will be compared with CPLEX, a classical solver. Both tools are deeply connected because the forecasted price of the different assets is used for the cost function construction.

<center><img src="./Images/Counselor1.png" width="400"></center>

## Portfolio optimization

For portfolio optimization, we use the modern portfolio theory where we is wanted to maximize the return of an investment while keeping the risk of losing money low. We based or method in the work of Mugel et. al [1] where the cost function is described by:

<center><img src="./Images/Cost-function.png" width="800"></center>


## Proposal 
This project uses different methods and techniques of Quantum computing and Quantum machine learning as:

- QNN,
- VQE,
- QAOA.

and classical methods as  ANN, LSTM, Optimizers (Cobyla) 


# References 
[1] Mugel, S., Kuchkovsky, C., Sánchez, E., Fernández-Lorenzo, S., Luis-Hita, J., Lizaso, E., & Orús, R. (2022). Dynamic portfolio optimization with real datasets using quantum processors and quantum-inspired tensor networks. Physical Review Research, 4(1), 1–13. https://doi.org/10.1103/PhysRevResearch.4.013006
[2] Chen, Samuel & Yoo, Shinjae & Fang, Yao-Lung. (2020). Quantum Long Short-Term Memory. 
[3] Bausch, Johannes. (2020). Recurrent Quantum Neural Networks. 
[4] Verdon, Guillaume & Broughton, Michael & Mcclean, Jarrod & Sung, Kevin & Babbush, Ryan & Jiang, Zhang & Neven, Hartmut & Mohseni, M.. (2019). Learning to learn with quantum neural networks via classical neural networks. 
