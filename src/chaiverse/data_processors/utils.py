import copy
from datasets import load_dataset


def get_raw_dataset(dataset_name, val_set_size=0.01, max_samples=-1):
    raw_datasets = load_dataset(dataset_name)
    max_samples = (
        min(max_samples, len(raw_datasets["train"]))
        if max_samples > 0
        else len(raw_datasets["train"])
    )
    if val_set_size > 0:
        if "validation" not in raw_datasets.keys():
            max_val = int(max_samples * val_set_size)
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                split=f"train[:{max_val}]",
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                split=f"train[{max_val}:{max_samples}]",
            )
        else:
            raw_datasets["train"] = raw_datasets["train"].select(range(max_samples))
    else:
        raw_datasets["train"] = load_dataset(
            dataset_name,
            split=f"train[:{max_samples}]",
        )
    return raw_datasets


def tokenize_function(examples, tokenizer, max_length):
    input_texts = examples['model_input']
    output_texts = examples['model_output']
    data = [input_ + output_ for input_, output_ in zip(input_texts, output_texts)]
    
    inputs = tokenizer(
        data,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_token_type_ids=False
    )

    inputs["labels"] = copy.deepcopy(inputs.input_ids)
    batch_size = len(inputs["labels"])
    return inputs


def disable_input_text_tokens(tokenizer, output_texts, inputs, batch_size):
    """We only want to train on the output_text tokens so set others to -100. """
    output_lengths = [len(tokenizer(output_string).input_ids) for output_string in output_texts]

    for batch in range(batch_size):
        num_input_tokens = len(inputs['labels'][batch]) - output_lengths[batch]
        for token in range(0, num_input_tokens):
            inputs["labels"][batch][token] = -100
    return inputs
