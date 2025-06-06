import torch
import tqdm
import wandb

def batch_log(args, step, loss, optimizer):
    if step % args.log_freq == 0:
        wandb.log({"train/loss": loss, "train/lr": optimizer.param_groups[0]['lr']})

def epoch_log(args, epoch, train_loss, train_acc, val_loss, val_acc):
    print(f"epoch [{epoch:2}/{args.num_epochs}] train_loss {train_loss:.4f} train_acc {train_acc:.2f}%")
    print(f"epoch [{epoch:2}/{args.num_epochs}] val_loss {val_loss:.4f} val_acc {val_acc:.2f}%")
    
    wandb.log({"train/epoch_loss": train_loss, 
               "train/epoch_acc": train_acc, 
               "val/epoch_loss": val_loss, 
               "val/epoch_acc": val_acc, 
               "epoch": epoch})

def train(args, epoch, loader, model, criterion, optimizer, scheduler):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0

    for step, (inputs, labels) in enumerate(tqdm.tqdm(loader, desc=f"Train epoch {epoch}")):
        inputs, labels = inputs.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        batch_log(args, step, loss.item(), optimizer)

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

def validate(args, epoch, loader, model, criterion):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader, desc=f"Val epoch {epoch}"):
            inputs, labels = inputs.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

class EarlyStopper:
    def __init__(self, model, model_dir, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.model = model
        self.model_dir = model_dir
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            print(f'best_val_loss {val_loss:.4f}, save model to {self.model_dir}!')
            torch.save(self.model.state_dict(), self.model_dir)
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
