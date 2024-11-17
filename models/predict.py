import torch
from transformers import BertTokenizerFast

from models.bert import BERTForQA, BERTModel

def predict_answer(question, context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    model = BERTForQA(
        BERTModel(30522, 768, 12, 12, 3072, 512)
    ).to(device)
    model.eval()

    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, token_type_ids)
        start_pred = torch.argmax(start_logits, dim=-1)
        end_pred = torch.argmax(end_logits, dim=-1)

    answer_tokens = inputs["input_ids"][0][start_pred:end_pred + 1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)
