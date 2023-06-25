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

def train_model(tokenizer, dataset, training_data_dir):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    config = tr.BertConfig(vocab_size = tokenizer.vocab_size)
    model = tr.BertForMaskedLM(config = config)
    
    training_args = tr.TrainingArguments(
        output_dir = training_data_dir,
        num_train_epochs = 1,
        save_steps = 5_000,
        save_total_limit = 2,
        prediction_loss_only = True,
        remove_unused_columns = False,
        per_device_train_batch_size = 10,
        logging_steps = 250,
        learning_rate = 1e-4,
        max_steps = 100_000,
        optim = "adamw_torch"
    )
    trainer = tr.Trainer(
        model = model,
        args = training_args,
        data_collator = tr.DataCollatorWithPadding(tokenizer = tokenizer, padding = "max_length", max_length = 512),
        train_dataset = dataset
    )
    trainer.train()

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
    dataset = ds.IterableDataset.from_generator(
        generator = su.parse_records, 
        gen_kwargs = { "directory": config["dataset_directory"].get(str) }
    )

    tokenized_dataset = tu.tokenize_dataset(tokenizer, dataset)
    masked_dataset = tokenized_dataset.map(lambda x: mask_data(x, tokenizer))

    train_model(tokenizer, masked_dataset, config["training_directory"].get(str))
    