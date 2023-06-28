import datasets as ds
import tokenizers as tk
import transformers as tr

def load_tokenizer(tokenizer_file):
    tokenizer = tr.PreTrainedTokenizerFast(tokenizer_file = tokenizer_file, return_special_tokens_mask = True)
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.unk_token = "[UNK]"
    return tokenizer

def prune_and_tokenize_dataset(sequences, tokenizer):
    del_idxs = []

    for idx, sequence in enumerate(sequences["input_ids"]):
        seq_len = len(sequence)
        X_count = sequence.count('X')
        if len(sequence) < 44:
            del_idxs.append(idx)
        elif len(sequence) > 512:
            del_idxs.append(idx)
        elif (X_count / seq_len) > 0.05:
            del_idxs.append(idx)
        
    sequences["input_ids"] = [sequence for idx, sequence in enumerate(sequences["input_ids"]) if idx not in del_idxs]
    sequences = tokenizer(sequences["input_ids"])
    return sequences