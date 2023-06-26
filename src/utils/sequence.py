import os
import math
import numpy

from Bio import SeqIO

rng = numpy.random.default_rng()

# lazily load sequences from a single fasta file
def parse_record(file):
    with open(file) as handle:
        for record in SeqIO.FastaIO.SimpleFastaParser(handle):
            yield record[1]
            
# lazily load sequences from a directory of fasta files
def parse_records(directory):
    for file in os.listdir(directory):
        if file.endswith(".fasta"):
            with open(os.path.join(directory, file)) as handle:
                for record in SeqIO.FastaIO.SimpleFastaParser(handle):
                    yield { "input_ids": record[1] }

def mask_data(sequences, tokenizer):
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")

    sequences["labels"] = []
    for sequence in sequences["input_ids"]:
        selection = rng.random()
        if selection <= 0.3:
            sequences["labels"].append(mask_multiple_blocks(sequence, mask_id, 0.15))
        else:
            sequences["labels"].append(mask_random(sequence, mask_id, 0.15))
    return sequences

# mask random elements of a sequence
def mask_random(sequence, mask, fraction):
    seq_len = len(sequence)
    labels = [-100] * seq_len
    seq_len -= 1
    n_mask = math.floor(seq_len * fraction)
    mask_idx = rng.choice(seq_len, n_mask, replace = False)
    mask_idx = [x + 1 for x in mask_idx]
    for idx in mask_idx:
        labels[idx] = sequence[idx]
        sequence[idx] = mask
    return labels

# randomly select a single block for masking
def mask_block(sequence, mask, fraction):
    seq_len = len(sequence)
    labels = [-100] * seq_len
    seq_len -= 1
    mask_len = math.floor(seq_len * fraction)
    mask_idx = rng.integers(0, seq_len - mask_len + 1)
    mask_idx = [x + 1 for x in mask_idx]
    labels[mask_idx:(mask_idx + mask_len)] = sequence[mask_idx:(mask_idx + mask_len)]
    sequence[mask_idx:(mask_idx + mask_len)] = [mask] * mask_len
    return labels

# randomly select multiple blocks for masking
def mask_multiple_blocks(sequence, mask, fraction):
    seq_len = len(sequence)
    labels = [-100] * seq_len
    seq_len -= 1
    block_len = rng.poisson(2.5) + 1
    n_mask = seq_len * fraction
    n_positions = round(n_mask / block_len)
    mask_idx = rng.choice(seq_len, n_positions, replace = False)
    mask_idx = [x + 1 for x in mask_idx]
    for idx in mask_idx:
        start = max(0, idx - (block_len // 2))
        end = min(seq_len, idx + (block_len // 2) + 1)
        labels[start:end] = sequence[start:end]
        sequence[start:end] = [mask] * (end - start)
    return labels

# split sequence into blocks and shuffle
def permute_blocks(sequence, min, max):
    n = rng.integers(min, max + 1)
    blocks = numpy.array_split(sequence, n)
    rng.shuffle(blocks)
    sequence = [item for sublist in blocks for item in sublist]

# truncate or pad sequence to seq_len
def normalize_sequence(sequence, token, output_len):
    seq_len = len(sequence)
    if seq_len > output_len:
        del sequence[output_len:]
    elif seq_len < output_len:
        sequence += [token] * (output_len - seq_len)