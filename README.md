# bitcoin-keras-nn-buy

Using a collection method with various API's, our cohort collected sequence level Bitcoin transactions. Then, using business logic, we implemented a "Buy","Sell", or "Wait" signal. The purpose of this neural network is to predict the binary outcome "Buy" or "Don't Buy". We plan on using other business logic to determine when to appropriately Sell. Below are a few notes on the modeling approach:

* Class imbalance (rare "Buy" outcomes) is managed with Oversampling the Minority Class
* Train/Test split of 70%/30%
* The optimizer used is Adam with a learning rate of 0.001
* The training job to get ~1M transactions spans ~2 weeks of Bitcoin transactions

## General Notes

I did not find TensorBoard particularly useful, but I can see it having value if you are tuning a whole bunch of jobs with different configurations. Having said that, using something like AWS Sagemaker would be a better approach. There are better techniques for remedy of class imbalance - these are not covered in this repo.

## Training Results

![alt text](https://github.com/datavizhokie/bitcoin-keras-nn-buy/blob/master/best_training_job.png)
