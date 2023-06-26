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