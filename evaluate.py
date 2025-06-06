import torch
import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_utils import get_test_loader
from model import CNN
import torch.nn as nn
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_path', type=str, default='./ckpt/model.pth')
    return parser.parse_args()

def test(args, loader, model, criterion):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader):
            inputs, labels = inputs.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            y_true.append(labels)
            y_pred.append(predicted)

    avg_loss = total_loss / total
    acc = 100. * correct / total

    y_true = torch.cat(y_true, dim=0).to('cpu').numpy()
    y_pred = torch.cat(y_pred, dim=0).to('cpu').numpy()
    
    return avg_loss, acc, y_true, y_pred

def main():
    args = parse_args()
    model = CNN().to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    criterion = nn.CrossEntropyLoss()

    test_loader, classes = get_test_loader(args)
    test_loss, test_acc, y_true, y_pred = test(args, test_loader, model, criterion)
    print(f"test_loss {test_loss:.4f} test acc {test_acc:.2f}%")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    class_report = classification_report(y_true, y_pred, output_dict=True)

    cm = confusion_matrix(y_true, y_pred)
    class_names = test_loader.dataset.classes
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with Labels')
    cm_figure_labeled = plt.gcf()
    plt.show()