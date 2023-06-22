import numpy
import collections
import transformers as tr
import sequence_utils as su

rng = numpy.random.default_rng()

def data_collator(features):
    for feature in features:
        token_ids = feature.pop("input_ids")
    selection = rng.random()
    

def train_model(tokenizer_file, tokenized_dataset):

    tokenizer = tr.PreTrainedTokenizerFast(tokenizer_file = tokenizer_file, return_special_tokens_mask = True)
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.cls_token = "[CLS]"

    config = tr.BertConfig(vocab_size = 16_384)
    model = tr.BertForMaskedLM(config = config)

    data_collator = tr.DataCollatorForLanguageModeling(
        tokenizer = tokenizer, 
        mlm = True, 
        mlm_probability = 0.15
    )
    training_args = tr.TrainingArguments(
        output_dir = "./training_data",
        overwrite_output_dir = True,
        num_train_epochs = 1,
        save_steps = 10_000,
        save_total_limit = 2,
        prediction_loss_only = True,
        remove_unused_columns = True,
        per_device_train_batch_size = 10,
        learning_rate = 1e-4
    )
    trainer = tr.Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = tokenized_dataset,
    )

    trainer.train()