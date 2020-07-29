# SeqGenSQL
A T5 based sequence generation model for WikiSQL task. Achieving 90.3% on test data set using sequence generation without logical form.

In this model, we experimented with following:

1. Feature Engineering 
- Adding Data Type to input
- Adding Data Samples to input

2. Data Augmentation
- Replacing Select column from training data
- Replacing Condition value for where clause

3. Reversed Trainer model
- Generate silver data for training purposes

4. Gated Extraction Network
- Modifed T5 and add a gate layer to decide whether a token should be extracted / generated. 
