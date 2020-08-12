from dataset import WikiSqlDataset
from model import LoggingCallback,SeqGenSQL
from dbengine_seqgen import DBEngine as db
import argparse
import os
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm                    
import urllib.request

# .\score.py --ckpt_download_url https://onebigdatabag.blob.core.windows.net/shared/base_epoch%3D12-val_loss%3D0.02616.ckpt --ckpt_path SeqGenSQL.ckpt
if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data")
    parser.add_argument('--data_type', default="dev", help="train|dev|test")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--base_model", default="t5-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt_download_url", default=None)
    parser.add_argument("--ckpt_path", default="SeqGenSQL.ckpt")
    parser.add_argument("--include_data_type", default=True)
    parser.add_argument("--num_sample_rows", type=int, default=3)
    parser.add_argument("--data_aug", default=[], help="List, use one of these options: ['select_column', 'where_value']. Default is []")
    parser.add_argument("--use_modified_network", default=False, help="Use gated layer to decide whether to extract or to generate")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=200)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--device", default="cuda", help="cpu|cuda")
    parser.add_argument("--silent", default=False, help="Output paramters")
    args = parser.parse_args()
  
    if args.num_return_sequences  > 1:
        args.batch_size = 1

    model_name = os.path.basename(args.ckpt_path)
    log_file = '{}/{}.score.log'.format(args.output_dir, model_name.replace("/","_"))

    if not args.silent:
        print("======================================================")
        print('Error log file name:', log_file)
        print("======================================================")
    
    if args.ckpt_download_url != None and not os.path.exists(args.ckpt_path):
        print("Downloading checkpoint...", end="")
        urllib.request.urlretrieve(args.ckpt_download_url, args.ckpt_path)
        print("Done!")

    print("Loading database file...", end="")
    dbeng = db('{}/{}.db'.format(args.data_dir, args.data_type))
    print("Done!")

    print("Loading T5FinalTuner pretrained model...", end="")
    model = SeqGenSQL.load_from_checkpoint(args.ckpt_path)
    print("Done!")

    if args.device == 'cuda':
        print("Loading model to cuda...", end="")
        model = model.to('cuda')
        print("Done!")


    print("Loading dataset...")
    dataset = WikiSqlDataset(model.tokenizer, 
        args.data_dir, 
        args.data_type,
        include_sample_data =args.num_sample_rows, 
        max_input_len=args.max_seq_length, 
        max_output_len=args.max_output_length,include_question = True)

    if args.num_return_sequences  > 1:
        args.batch_size = 1

    # generate sql statement
    print("Generating sequences...", end="")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    outputs = []
    targets = []
    di = 0
    for batch in tqdm(loader):

        if args.num_return_sequences == 1:
            input_ids = batch['source_ids']
            attention_mask = batch['source_mask']
        else:
            # input_ids = torch.unsqueeze(batch['source_ids'],0)
            # attention_mask = torch.unsqueeze(batch['source_mask'],0)
            input_ids = batch['source_ids']
            attention_mask = batch['source_mask']


        if args.device == 'cuda':
            outs = model.model.generate(input_ids=input_ids.cuda(), 
                                        attention_mask=attention_mask.cuda(), 
                                        num_beams = args.num_return_sequences, 
                                        max_length=args.max_output_length,
                                        num_return_sequences = args.num_return_sequences)
        else:
            outs = model.model.generate(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            max_length=args.max_output_length,
                            num_beams = args.num_return_sequences, 
                            num_return_sequences = args.num_return_sequences)

        if args.num_return_sequences > 1:
            guided_out = model.tokenizer.decode(outs[0])
            target = model.tokenizer.decode(batch["target_ids"][0])
            execution_failed = False
            for i, beam_output in enumerate(outs):
                try:
                    dec = model.tokenizer.decode(beam_output)
                    if execution_failed:
                        print("  ",dec)
                    pred_lf = dbeng.generate_logical_form(model.tokenizer, dec,batch['question'][0],
                                                        dataset.tables,
                                                        dataset.agg_ops, dataset.cond_ops,"sql")                    
                    pred_result = dbeng.execute_query(table_id = pred_lf["table_id"], query = pred_lf)
                    guided_out = dec
                    break               
                except:
                    print("====================================================")
                    print("question {}: {}".format(di, model.tokenizer.decode(batch['source_ids'][0]).replace('⁇','<')))
                    #print(d)
                    print("  True:", target.replace('⁇','<'))
                    print("  Pred:", dec.replace('⁇','<'))
                    print("  lf:",pred_lf)
                    execution_failed = True
            di += 1
            outputs.append(guided_out)
            targets.append(target)
        else:
            guided_out = [model.tokenizer.decode(ids) for ids in outs]
            target = [model.tokenizer.decode(ids) for ids in batch["target_ids"]]
            outputs.extend(guided_out)
            targets.extend(target)
    print("Done!")

    # Score
    print("Scoring...", end="")
    f = open(log_file ,"w")
    correct = 0
    incorrect = {"sel_neg":0, "sel_mismatch":0,"agg_neg":0, "agg_mismatch":0, "cond_col_neg":0,"cond_col_mismatch":0,
                "cond_op_neg":0, "cond_op_mismatch":0,"cond_val_neg":0,"cond_val_mismatch":0}
    for t,d, i in zip(targets, outputs, dataset.data):
        try:
            if t == d:
                correct += 1
            else:
                pred_lf = dbeng.generate_logical_form(model.tokenizer, d,i['question'],
                                                      dataset.tables,
                                                      dataset.agg_ops, dataset.cond_ops,"sql")
                if pred_lf['sql']!={}:
                    pred_result = dbeng.execute_query(table_id = pred_lf["table_id"], query = pred_lf)
                else:
                    f.write("===================== ERROR ========================\n")
                    f.write("Question: {}\n".format(i['question']))
                    f.write("Pred: {} lf: {} RESULT: {}\n".format(d,pred_lf['sql'],pred_result))
                    f.write("True: {} lf: {} RESULT: {}\n\n".format(t, i['sql'], true_result))
                    continue

                true_result = dbeng.execute_query(table_id = i["table_id"], query = i)
                if (pred_result == true_result):
                    correct += 1
                else:
                    f.write("===================== ERROR ========================\n")
                    f.write("Question: {}\n".format(i['question']))
                    f.write("Pred: {} lf: {} RESULT: {}\n".format(d,pred_lf['sql'],pred_result))
                    f.write("True: {} lf: {} RESULT: {}\n\n".format(t, i['sql'], true_result))
                    #print(pred_lf['sql']['sel'])
                    if (pred_lf['sql']['sel'] != i['sql']['sel']):
                        incorrect['sel_mismatch'] += 1
                    if (pred_lf['sql']['agg'] != i['sql']['agg']):
                        incorrect['agg_mismatch'] += 1
                    for c in pred_lf['sql']['conds']:
                        if c[0] == -1:
                            incorrect['cond_col_neg'] += 1
                        if c[1] == -1:
                            incorrect['cond_cop_neg'] += 1
                        if i['question'].find(str(c[2])) == -1:
                            incorrect['cond_val_neg'] += 1

        except:
            f.write("===================== ERROR ========================\n")
            f.write("Question: {}\n".format(i['question']))
            f.write("Pred: {} lf: {}\n".format(d,pred_lf['sql']))
            f.write("True: {} lf: {}\n\n".format(t, i['sql']))

            #print(pred_lf['sql']['sel'])
            if (pred_lf['sql']['sel'] == -1):
                incorrect['sel_neg'] += 1
            if (pred_lf['sql']['agg'] == -1):
                incorrect['agg_neg'] += 1
            for c in pred_lf['sql']['conds']:
                if c[0] == -1:
                    incorrect['cond_col_neg'] += 1
                if c[1] == -1:
                    incorrect['cond_op_neg'] += 1
                if i['question'].find(str(c[2])) == -1:
                    incorrect['cond_val_neg'] += 1

    f.close()
    print(incorrect)
    print("Correct: {} Total: {} Ratio:{:.5f} ".format(correct, len(targets), correct/len(targets)))
    print("All Completed!")
