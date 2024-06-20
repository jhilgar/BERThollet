import numpy
import confuse
import pathlib
import argparse

import datasets as ds
import transformers as tr
import utils.token as tu
import utils.sequence as su

rng = numpy.random.default_rng()

def train_model(tokenizer, dataset, training_directory, args):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    config = tr.DistilBertConfig(vocab_size = tokenizer.vocab_size)
    model = tr.DistilBertForMaskedLM(config = config)
    
    training_args = tr.TrainingArguments(training_directory, **args)
    trainer = tr.Trainer(
        model = model,
        args = training_args,
        data_collator = tr.DataCollatorForTokenClassification(tokenizer = tokenizer, padding = "longest"),
        train_dataset = dataset
    )
    trainer.train()

if __name__ == "__main__":
    project_dir = pathlib.Path(__file__).parent.parent

    default_tokenizer_file = str(project_dir / "defaults/bpe_512.json")

    parser = argparse.ArgumentParser(description = "BERThollet model training script")
    parser.add_argument('--tokenizer_file', nargs = '?', default = default_tokenizer_file, type = str, help = "Path to a tokenizer file json")
    args = parser.parse_args()

    config = confuse.Configuration("BERThollet", __name__)
    config.set_file(project_dir / "defaults/config.yaml")
    
    tokenizer = tu.load_tokenizer(args.tokenizer_file)
    
    dataset = ds.load_dataset("jhilgar/uniparc", split = "train", streaming = True)
    dataset = dataset.map(lambda x: tokenizer(x["input_ids"], truncation = True, max_length = 512), batched = True, batch_size = 2000)
    dataset = dataset.map(lambda x: su.mask_data(x, tokenizer), batched = True, batch_size = 2000)
    train_model(tokenizer, dataset, config["training_directory"].get(str), config["trainingarguments"].get())