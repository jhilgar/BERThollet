import sequence_utils as su
import token_utils as tu
import train

filename = "300k.fasta"
records = su.parse_records(filename)
records = {
    "input_ids": range(0, len(records)),
    "sequences": records
}

#tu.tokenize_data("tokenizer", records, "tokenized_data")
tokenized_dataset = tu.load_tokenized_data("tokenized_data")
#tokenizer  = tu.load_tokenizer("tokenizer")
#tokenized_data = tu.tokenize_data(tokenizer, records, filename + ".tokenized")
#print(tokenized_dataset[0])
#tokenized_data = tu.tokenize_data(records)
#tokenizer = tu.train_tokenizer2(records, "tokenizer.json")

#print(records.values())
train.train_model(tokenized_dataset)