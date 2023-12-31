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
    parser.add_argument('fasta_file', type = str, help = "The path to a fasta file on which to train the tokenizer")
    args = parser.parse_args()
    
    config = confuse.Configuration("BERThollet", __name__)
    config.set_file(project_dir / "defaults/config.yaml")
    
    unknown_token = "[UNK]"
    special_tokens = ["[CLS]", "[MASK]", "[PAD]", unknown_token]
    if config["token_model"].get(str) == "wordpiece":
        tokenizer = tk.Tokenizer(tk.models.WordPiece(max_input_chars_per_word = 1, unk_token = unknown_token))
        trainer = tk.trainers.WordPieceTrainer(vocab_size = config["vocab_size"].get(int), special_tokens = special_tokens)
    else:
        tokenizer = tk.Tokenizer(tk.models.BPE(unk_token = unknown_token))
        trainer = tk.trainers.BpeTrainer(vocab_size = config["vocab_size"].get(int), special_tokens = special_tokens)

    sequences = su.parse_record(args.fasta_file)

    characters_to_remove = ['B', 'J', 'O', 'U', 'X', 'Z']
    translation_table = {ord(x): '' for x in characters_to_remove}

    sequences = map(lambda x: x.translate(translation_table), sequences)

    tokenizer.train_from_iterator(iterator = sequences, trainer = trainer)
    
    cls_token_id = tokenizer.token_to_id("[CLS]")
    tokenizer.post_processor = tk.processors.TemplateProcessing(
        single = "[CLS] $A",
        special_tokens = [("[CLS]", cls_token_id)]
    )
    
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
