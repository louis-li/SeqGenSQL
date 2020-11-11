# SeqGenSQL
Paper: http://arxiv.org/abs/2011.03836
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

## Train
python ./train.py


## Score
To score model, run:

python ./score.py --ckpt_download_url https://onebigdatabag.blob.core.windows.net/shared/base_gated_e09_0.02626.ckpt 

To score test data set, run:

python ./score.py --ckpt_download_url https://onebigdatabag.blob.core.windows.net/shared/base_gated_e09_0.02626.ckpt --data_type test

Score.py also generate an error log for failed prediction for further analysis (Logical form from prediction is generated after prediction for execution purposes)

===================== ERROR ========================

Question: What is the English name of the country whose official native language is Dutch Papiamento?

Pred: select [country ( endonym )] from [1-1008653-1] where [official or native language(s) (alphabet/script)] = 'dutch papiamento' lf:{'sel': 2, 'agg': 0, 'conds': [[4, 0, 'Dutch Papiamento']], 'where_value_idx': [[74]]} RESULT: [('aruba aruba',)] 

True: select [country ( exonym )] from [1-1008653-1] where [official or native language(s) (alphabet/script)] = 'dutch papiamento' lf: {'sel': 0, 'conds': [[4, 0, 'Dutch Papiamento']], 'agg': 0} RESULT: [('aruba',)] 


## Parameters

### Training Parameters:
--data_dir: train/dev/test data folder

--default_root_dir: folder to store checkpoints

--model_name_or_path: base model, t5-small/t5-base

--max_seq_lenght: length of input tokens,default 512

--max_output_length: length of output sequence, default 200

--learning_rate: default 2e-4

--num_train_epochs: default 25

--gpus: GPU to use, it could be a list of GPUs like [0,1] or -1 to use all GPUs, or 1 to use 1 GPU

--include_data_type: Include data types in input tokens, default yes

--num_sample_rows: Number of sample data included in input token, default 3

--data_aug: Data augementation options, could be 'select_column' and/or 'where_value'. Default []

--generated_data_files: Filenames for generate training data to be includeded, a list of filenames: ["datagen/e20_1.jsonl"]. Default []

--use_modified_network: Use gated extraction network. Default True. False means using original T5 implementation. 

### Scoring Parameters:
--data_dir: train/dev/test data folder

--data_type: dev or test. To score either dev or test data set

--base_model: t5-small/t5-base

--batch_size: Default 32

--ckpt_download_url: URL to download checkpoint file. Default None

--ckpt_path: Checkpoint filename. If ckpt_download_url is not None, ckpt_path will be filename to save checkpoint to

--include_data_type: Include data types in input tokens, default yes

--num_sample_rows: Number of sample data included in input token, default 3

--data_aug: Data augementation options, could be 'select_column' and/or 'where_value'. Default []

--use_modified_network: Use gated extraction network. Default True. False means using original T5 implementation

--num_return_sequences: Number of return sequences using Beam Search. Default 1



