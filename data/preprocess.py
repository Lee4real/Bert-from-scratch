from datasets import load_dataset
from transformers import BertTokenizerFast

# Load SQuAD dataset
def load_and_preprocess_data():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_dataset("squad")

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=512,
            padding="max_length",
            return_offsets_mapping=True,
        )
        start_positions, end_positions = [], []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            answer = examples["answers"][i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = tokenized.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - sequence_ids[::-1].index(1)

            token_start = token_end = tokenizer.cls_token_id
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    token_start = idx
                if start < end_char <= end:
                    token_end = idx
                    break

            if token_start < context_start or token_end >= context_end:
                token_start = token_end = tokenizer.cls_token_id
            start_positions.append(token_start)
            end_positions.append(token_end)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized.pop("offset_mapping", None)
        return tokenized

    return dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
