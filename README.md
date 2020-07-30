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


To score model, run:
python ./score.py --ckpt_download_url https://onebigdatabag.blob.core.windows.net/shared/base_epoch%3D12-val_loss%3D0.02616.ckpt

Score.py also generate an error log for failed prediction for further analysis

===================== ERROR ========================

Question: What is the English name of the country whose official native language is Dutch Papiamento?

Pred: select [country ( endonym )] from [1-1008653-1] where [official or native language(s) (alphabet/script)] = 'dutch papiamento' lf:{'sel': 2, 'agg': 0, 'conds': [[4, 0, 'Dutch Papiamento']], 'where_value_idx': [[74]]} RESULT: [('aruba aruba',)] 

True: select [country ( exonym )] from [1-1008653-1] where [official or native language(s) (alphabet/script)] = 'dutch papiamento' lf: {'sel': 0, 'conds': [[4, 0, 'Dutch Papiamento']], 'agg': 0} RESULT: [('aruba',)] 


