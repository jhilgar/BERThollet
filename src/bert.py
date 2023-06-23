import os
import pathlib

import sequence_utils as su
import token_utils as tu
import train
import datasets as ds

project_dir = pathlib.Path(__file__).parent.parent

data_dir = project_dir / "data/"
records_dir = data_dir / "sequences/"
tokenizer_file = data_dir / "tokenizer.json"
token_dir = data_dir / "tokens/"
training_data_dir = data_dir / "training/"

#tu.train_tokenizer(records, str(tokenizer_file))

tokenizer = tu.load_tokenizer(str(tokenizer_file))

dataset = ds.IterableDataset.from_generator(
    generator = su.parse_records, 
    gen_kwargs = { "directory": str(records_dir) }
)
tokenized_dataset = dataset.map(lambda x: tokenizer(x["input_ids"], padding = "max_length", truncation = True, max_length = 512), batched = True)

train.train_model(tokenizer, tokenized_dataset, training_data_dir)

