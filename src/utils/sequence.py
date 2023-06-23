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

# mask random elements of a sequence
def mask_random(sequence, mask, fraction):
    seq_len = len(sequence)
    n_mask = math.floor(seq_len * fraction)
    mask_idx = rng.choice(seq_len, n_mask, replace = False)
    for idx in mask_idx:
        sequence[idx] = mask

# randomly select a single block for masking
def mask_block(sequence, mask, fraction):
    seq_len = len(sequence)
    mask_len = math.floor(seq_len * fraction)
    mask_idx = rng.integers(0, seq_len - mask_len + 1)
    sequence[mask_idx:(mask_idx + mask_len)] = [mask] * mask_len

# randomly select multiple blocks for masking
def mask_multiple_blocks(sequence, mask, fraction):
    seq_len = len(sequence)
    block_len = rng.poisson(2.5) + 1
    n_mask = seq_len * fraction
    n_positions = round(n_mask / block_len)
    mask_idx = rng.choice(seq_len, n_positions, replace = False)
    for idx in mask_idx:
        start = max(0, idx - (block_len // 2))
        end = min(seq_len, idx + (block_len // 2) + 1)
        sequence[start:end] = [mask] * (end - start)

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