import yaml
import argparse

import utils.sequence as su
import utils.token as tu
import tokenizers as tk
import datasets as ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "BERThollet tokenizer training script")
    parser.add_argument(
        'config_file', 
        type = str,
        help = 'The path to a config yaml')
    args = parser.parse_args()

    with open(args.config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())

    special_tokens = ["[CLS]", "[MASK]", "[PAD]"]
    if config["token_model"] == "wordpiece":
        tokenizer = tk.Tokenizer(tk.models.WordPiece(max_input_chars_per_word = 1))
        trainer = tk.trainers.WordPieceTrainer(vocab_size = config["vocab_size"], special_tokens = special_tokens)
    else:
        tokenizer = tk.Tokenizer(tk.models.BPE())
        trainer = tk.trainers.BpeTrainer(vocab_size = config["vocab_size"], special_tokens = special_tokens)

    sequences = su.parse_record(config["token_training_records"])
    tokenizer.train_from_iterator(sequences, trainer = trainer)
    tokenizer.save(config["tokenizer_file"])