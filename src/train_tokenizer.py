import pathlib
import confuse
import argparse

import utils.sequence as su
import utils.token as tu
import tokenizers as tk
import datasets as ds

if __name__ == "__main__":
    project_dir = pathlib.Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description = "BERThollet tokenizer training script")
    parser.add_argument(
        'fasta_file',
        type = str,
        help = "The path to a fasta file on which to train the tokenizer"
    )
    parser.add_argument(
        '--config', 
        type = str,
        help = 'The path to an optional config yaml')
    args = parser.parse_args()
    
    config = confuse.Configuration("BERThollet", __name__)
    config.set_file(project_dir / "default.yaml")
    
    unknown_token = "[UNK]"
    special_tokens = ["[CLS]", "[MASK]", "[PAD]", unknown_token]
    if config["token_model"] == "wordpiece":
        tokenizer = tk.Tokenizer(tk.models.WordPiece(max_input_chars_per_word = 1))
        trainer = tk.trainers.WordPieceTrainer(vocab_size = config["vocab_size"], special_tokens = special_tokens)
    else:
        tokenizer = tk.Tokenizer(tk.models.BPE())
        trainer = tk.trainers.BpeTrainer(vocab_size = config["vocab_size"].get(int), special_tokens = special_tokens)

    sequences = su.parse_record(args.fasta_file)
    tokenizer.train_from_iterator(sequences, trainer = trainer)

    fasta_file = pathlib.Path(args.fasta_file).name

    output_filename = str(
        project_dir / "data" / "{}_{}_{}.json".format(
            config["token_model"], 
            fasta_file, 
            config["vocab_size"]
        )
    )
    tokenizer.save(output_filename)
    print("Tokenizer output saved to {}".format(output_filename))
