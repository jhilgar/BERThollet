import numpy
import confuse
import pathlib
import argparse

import datasets as ds
import transformers as tr
import utils.token as tu
import utils.sequence as su

rng = numpy.random.default_rng()

def mask_data(sequence, tokenizer):
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    cls_id = tokenizer.convert_tokens_to_ids("[CLS]")

    selection = rng.random()
    if selection <= 0.3:
        sequence["labels"] = su.mask_multiple_blocks(sequence["input_ids"], mask_id, 0.15)
    else:
        sequence["labels"] = su.mask_random(sequence["input_ids"], mask_id, 0.15)
    return sequence

def train_model(tokenizer, dataset, training_directory, args):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    config = tr.BertConfig(vocab_size = tokenizer.vocab_size)
    model = tr.BertForMaskedLM(config = config)
    
    training_args = tr.TrainingArguments(training_directory, **args)
    trainer = tr.Trainer(
        model = model,
        args = training_args,
        data_collator = tr.DataCollatorWithPadding(tokenizer = tokenizer, padding = "max_length", max_length = 512),
        train_dataset = dataset
    )
    trainer.train(resume_from_checkpoint = True)

if __name__ == "__main__":
    project_dir = pathlib.Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description = "BERThollet model training script")
    parser.add_argument(
        'tokenizer_file',
        type = str,
        help = "The path to a tokenizer file json"
    )
    parser.add_argument(
        '--config', 
        type = str,
        help = 'The path to an optional config yaml')
    args = parser.parse_args()

    config = confuse.Configuration("BERThollet", __name__)
    config.set_file(project_dir / "default.yaml")
    
    tokenizer = tu.load_tokenizer(args.tokenizer_file)
    
    dataset = ds.load_dataset("jhilgar/uniparc", split = "train", streaming = True)
    tokenized_dataset = tu.tokenize_dataset(tokenizer, dataset)
    masked_dataset = tokenized_dataset.map(lambda x: mask_data(x, tokenizer))
    train_model(tokenizer, masked_dataset, config["training_directory"].get(str), config["trainingarguments"].get())