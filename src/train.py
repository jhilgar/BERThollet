import transformers as tr
import datasets as ds

def train_model(tokenizer_file, tokenized_dataset):

    #fix tokenizer json
    tokenizer = tr.PreTrainedTokenizerFast(tokenizer_file = tokenizer_file)
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.cls_token = "[CLS]"

    config = tr.BertConfig(vocab_size = 512)
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
        remove_unused_columns = True
    )
    trainer = tr.Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = tokenized_dataset,
    )
    trainer.train()