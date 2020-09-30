# bitcoin-keras-nn-buy

Using a collection method with various API's, our cohort collected sequence level Bitcoin transactions. Then, using business logic, we implemented a "Buy","Sell", or "Wait" signal. The purpose of this neural network is to predict the binary outcome "Buy" or "Don't Buy". We plan on using other business logic to determine when to appropriately Sell. Below are a few notes on the modeling approach:

* Class imbalance (rare "Buy" outcomes) is managed with Oversampling the Minority Class
* Train/Test split of 70%/30%
* The optimizer used is Adam with a learning rate of 0.001
* The sample pulled from Snowflake has 825k transactions spanning 2020-09-15 through 2020-09-24 (after oversampling the training data, there are ~1.4M records) 

Below is a snapshot of the transaction data:

![alt text](https://github.com/datavizhokie/bitcoin-keras-nn-buy/blob/master/data_snapshot.png)

## General Notes

I did not find TensorBoard particularly useful, but I can see it having value if you are tuning a whole bunch of jobs with different configurations. Having said that, using something like AWS Sagemaker would be a better approach. There are better techniques for remedy of class imbalance - these are not covered in this repo.

## Training Results

The bulk of HP updates were around attempting to drive down the False Positive Rate. From a business use case perspective, falsely saying that a "Buy" signal occurs is not really falsely saying the buyer would lose profit on account of two reasons:

1. Profit is certainly determined by the strategy behind "Sell"
2. The Buy/Sell/Wait signal is a heuristic in the first place, meaning that it should loosely be considered a "ground truth"

![alt text](https://github.com/datavizhokie/bitcoin-keras-nn-buy/blob/master/best_training_job.png)


**The crypto Bitcoin data (not supplied) is proprietary to Finditly, LLC. **


