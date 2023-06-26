import pathlib
import argparse

from Bio import SeqIO
import datasets as ds
import utils.sequence as su

if __name__ == "__main__":
    project_dir = pathlib.Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description = "BERThollet sequence data uploader")
    parser.add_argument(
        'fasta_dir',
        type = str,
        help = "The path to fasta files to be uploaded"
    )
    args = parser.parse_args()

    dataset = ds.Dataset.from_generator(
        generator = su.parse_records, 
        gen_kwargs = { "directory": args.fasta_dir }
    )

    dataset.push_to_hub("jhilgar/uniparc")
    