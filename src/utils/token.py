import datasets as ds
import tokenizers as tk
import transformers as tr

def train_tokenizer(sequence, filename):
    tokenizer = tk.Tokenizer(tk.models.BPE())
    special_tokens = ["[CLS]", "[MASK]", "[PAD]"]
    trainer = tk.trainers.BpeTrainer(vocab_size = 16_384, special_tokens = special_tokens)
    tokenizer.train_from_iterator(sequence, trainer = trainer)
    
    tokenizer.save(filename)
    return tokenizer

def load_tokenizer(tokenizer_file):
    tokenizer = tr.PreTrainedTokenizerFast(tokenizer_file = tokenizer_file)
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.cls_token = "[CLS]"
    return tokenizer

def tokenize_dataset(tokenizer, dataset):
    tokenized_dataset = dataset.map(
    lambda x: tokenizer(
        x["input_ids"], 
        padding = "max_length", 
        truncation = True, 
        max_length = 512
        ), 
    batched = True
    )
    return tokenized_dataset