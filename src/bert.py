import sequence_utils as su
import token_utils as tu
import train

records_file = "../data/300k.fasta"
tokenizer_file = "../data/tokenizer.json"
token_dir = "../data/tokens"
'''
records = su.parse_records(records_file)
tokenizer = tu.train_tokenizer(records, tokenizer_file)

tokenizer  = tu.load_tokenizer(tokenizer_file)
records = {
    "input_ids": records
}
tu.tokenize_data(tokenizer_file, records, token_dir)
'''
tokenized_dataset = tu.load_tokenized_data(token_dir)

train.train_model(tokenizer_file, tokenized_dataset)