import numpy
import argparse
import collections

import datasets as ds
import transformers as tr
import utils.token as tu
import utils.sequence as su

rng = numpy.random.default_rng()

def data_collator_rgn2(features, tokenizer):
    for feature in features:
        selection = rng.random()
        su.mask_block(feature["input_ids"], 1, 0.2)

    return tr.default_data_collator(features)

def train_model(tokenizer, dataset, training_data_dir):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    config = tr.BertConfig(vocab_size = 16_384)
    model = tr.BertForMaskedLM(config = config)

    data_collator = tr.DataCollatorForLanguageModeling(
        tokenizer = tokenizer, 
        mlm = True, 
        mlm_probability = 0.15
    )
    training_args = tr.TrainingArguments(
        output_dir = training_data_dir,
        overwrite_output_dir = True,
        num_train_epochs = 1,
        save_steps = 10_000,
        save_total_limit = 2,
        prediction_loss_only = True,
        remove_unused_columns = False,
        per_device_train_batch_size = 10,
        learning_rate = 1e-4,
        max_steps = 100_000,
        optim = "adamw_torch"
    )
    trainer = tr.Trainer(
        model = model,
        args = training_args,
        #data_collator = lambda a: data_collator_rgn2(a, tokenizer),
        #data_collator = tr.default_data_collator,
        data_collator = data_collator,
        train_dataset = dataset,
    )
    '''
    data_collator = trainer.get_train_dataloader().collate_fn
    actual_train_set = trainer._remove_unused_columns(trainer.train_dataset)
    batch = data_collator([actual_train_set[0]])
    print(batch)
    '''
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "BERThollet model training script")
    parser.add_argument(
        'tokenizer_file', 
        type = str,
        help = 'The path to a tokenizer json file')
    parser.add_argument(
        'dataset_dir', 
        type = str,
        help = 'The path to a directory containing fasta files')
    parser.add_argument(
        'training_data_dir', 
        type = str,
        help = 'The path to a directory where training files will be out')
    args = parser.parse_args()

    tokenizer = tu.load_tokenizer(args.tokenizer_file)
    dataset = ds.IterableDataset.from_generator(
        generator = su.parse_records, 
        gen_kwargs = { "directory": args.dataset_dir }
    )
    tokenized_dataset = tu.tokenize_dataset(tokenizer, dataset)
    train_model(tokenizer, tokenized_dataset, args.training_data_dir)