import numpy
import confuse
import pathlib
import argparse

import torch
import datasets as ds
import accelerate as ac
import transformers as tr
import evaluate as ev
import utils.token as tu
import utils.sequence as su

rng = numpy.random.default_rng()

class AminoBERT(torch.nn.Module):
    def __init__(self, config):
        super(AminoBERT, self).__init__()

        self.classifier = torch.nn.Linear(768, 2)
        self.dropout = torch.nn.Dropout(0.1)
        self.model = tr.DistilBertForMaskedLM(config = config)

    def forward(self, input_ids, attention_mask, labels, permutes = None):
        output = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, return_dict = True)

        sequence_output = self.dropout(output["hidden_states"][0])
        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))
        
        return output

def train_model(tokenizer, dataset, training_directory, args):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    dataset = dataset.remove_columns("token_type_ids")

    accelerator = ac.Accelerator(mixed_precision = "fp16")
    collator = tr.DataCollatorForTokenClassification(tokenizer = tokenizer, padding = "longest")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 8, collate_fn = collator)
    config = tr.DistilBertConfig(vocab_size = tokenizer.vocab_size, output_hidden_states = True, output_attentions = True)
    model = AminoBERT(config = config)
    optimizer = torch.optim.AdamW(params = model.parameters())
    
    model, optimizer, data = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(10):
        for step, batch in enumerate(data):
            optimizer.zero_grad()
            output = model(**batch)
            accelerator.backward(output.loss)
            optimizer.step()
    
            if step % 250 == 0:
                print(loss)

if __name__ == "__main__":
    project_dir = pathlib.Path(__file__).parent.parent

    default_tokenizer_file = str(project_dir / "defaults/bpe_512.json")

    parser = argparse.ArgumentParser(description = "BERThollet model training script")
    parser.add_argument('--tokenizer_file', nargs = '?', default = default_tokenizer_file, type = str, help = "Path to a tokenizer file json")
    args = parser.parse_args()

    config = confuse.Configuration("BERThollet", __name__)
    config.set_file(project_dir / "defaults/config.yaml")
    
    tokenizer = tu.load_tokenizer(args.tokenizer_file)
    
    dataset = ds.load_dataset("jhilgar/uniparc", split = "train", streaming = True)
    dataset = dataset.map(lambda x: tu.prune_and_tokenize_dataset(x, tokenizer), batched = True, batch_size = 2000)
    dataset = dataset.map(lambda x: su.mask_data(x, tokenizer), batched = True, batch_size = 2000)
    print(next(iter(dataset)))
    #train_model(tokenizer, dataset, config["training_directory"].get(str), config["trainingarguments"].get())