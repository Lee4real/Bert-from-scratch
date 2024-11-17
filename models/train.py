import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from models.bert import BERTForQA, BERTModel
from data.preprocess import load_and_preprocess_data

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_preprocess_data()

    # Prepare DataLoaders
    train_data = data["train"][:10000]
    valid_data = data["validation"][:1000]
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=8)

    # Model and optimizer setup
    model = BERTForQA(
        BERTModel(30522, 768, 12, 12, 3072, 512)
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(5):
        model.train()
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            optimizer.zero_grad()
            with autocast():
                start_logits, end_logits = model(input_ids, token_type_ids)
                start_loss = loss_fn(start_logits, start_positions)
                end_loss = loss_fn(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch + 1}: Loss = {total_loss.item():.4f}")
