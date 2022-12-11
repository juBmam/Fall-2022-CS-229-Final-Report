# CSDI
This is the github repository for Goutam Tulsiyan and Julian Cheng's Fall 2022 CS229 Final Project Report.  
In this project, we researched model studied in  CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation, found at https://arxiv.org/abs/2107.03502.

## Requirement
Please install the packages in requirements.txt



## Experiments 

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```


### Visualize results
'In the terminal .ipynb' is a notebook for visualizing results.

## Acknowledgements

This code is based on [BRITS](https://github.com/caow13/BRITS), [DiffWave](https://github.com/lmnt-com/diffwave), and [Ermongroup] (https://github.com/ermongroup/CSDI)
