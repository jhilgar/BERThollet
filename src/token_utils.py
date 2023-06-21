import datasets as ds
import tokenizers as tk
import transformers as tr

def load_tokenized_data(path):
    return ds.load_from_disk(path)
    
def train_tokenizer(sequence, filename):
    tokenizer = tk.Tokenizer(tk.models.BPE())
    special_tokens = ["[CLS]", "[MASK]", "[PAD]"]
    trainer = tk.trainers.BpeTrainer(vocab_size = 16_384, special_tokens = special_tokens)
    tokenizer.train_from_iterator(sequence, trainer = trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    tokenizer.post_processor = tk.processors.TemplateProcessing(
        single = f"[CLS]:0 $A:0",
        special_tokens = [
            ("[CLS]", cls_token_id)
        ]
    )
    tokenizer.save(filename)
    return tokenizer

def tokenize_data(tokenizer_file, sequences, output_file):
    tokenizer = tr.PreTrainedTokenizerFast(tokenizer_file = tokenizer_file)
    tokenizer.pad_token = "[PAD]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.mask_token = "[MASK]"
    
    dataset = ds.Dataset.from_dict(sequences)
    tokenized_dataset = dataset.map(lambda data: tokenizer(data["input_ids"], padding = "max_length", truncation = True, max_length = 512), batched = True)
    tokenized_dataset.save_to_disk(output_file)
    return tokenized_dataset

def load_tokenizer(filename):
    tokenizer = tk.Tokenizer.from_file(filename)
    return tokenizer