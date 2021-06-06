## ETB Modules/Use-Cases

. Credit Card

. Current Account

. Funds

. Insurance

. Loan

. Savings Account

. Term Deposit


## Basic Terminologies /  Modules used     

. Dask for Parallel Processing

. Every Use case has a up-sell / cross-sell module

. Pre-processing Pipeline

. Model Traininig Pipeline


## Dask for Parallel Processing

. Dask is a parallel computation framework that has seamless integration with your Jupyter notebook. Originally, it was built to overcome the storage limitations of a single machine and extend the computation capability of Pandas, Numpy, and Scit-kit Learn with DASK equivalents, but soon it found its use as a generic distributed system.



. Advantages of Dask: 


**Scalability**

Dask scales up Pandas, Scikit-Learn, and Numpy natively with python and Runs resiliently on clusters with multiple cores or can also be scaled down to a single machine.

**Scheduling**

Dask Task Schedulers are optimized for computation much like Airflow, Luigi. It provides rapid feedback, tracks tasks using Task graphs, and aids in diagnostics both in local and distributed mode aking it interactive and responsive.

![alt text](https://miro.medium.com/max/700/1*0OaznYUVfHwDJacwqfSBFg.png)


. Installation of Dask:

**Conda Installation**

```
conda install dask
```

**Pip Installation**

```
python -m pip install dask
```

. Conclusion:

Dask is a fault-tolerant, elastic framework for parallel computation in python that can be deployed locally, on the cloud, or high-performance computers. Not only it scales out capabilities of Pandas and NumPy, but also it can be used as Task schedulers. 
