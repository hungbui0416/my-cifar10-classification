import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from model import CNN
from train_utils import train, validate, epoch_log, EarlyStopper
from data_utils import get_train_val_loaders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing scheduler')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--wandb_entity', type=str, default='project-2')
    parser.add_argument('--wandb_project', type=str, default='CIFAR10-classification')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.1, help='early stopping min delta')
    parser.add_argument('--wandb_api_key', type=str, default='')
    return parser.parse_args()

def main():
    args = parse_args()

    train_loader, val_loader = get_train_val_loaders(args)

    model = CNN().to(args.device)
    model_dir = './ckpt/model.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None
    early_stopper = EarlyStopper(model, model_dir, patience=args.patience, min_delta=args.min_delta)

    wandb.login(key=args.wandb_api_key)
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq) 

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        val_loss, val_acc = validate(args, epoch, val_loader, model, criterion)

        epoch_log(args, epoch, train_loss, train_acc, val_loss, val_acc)

        if early_stopper.early_stop(val_loss):
            print('early stop at epoch', epoch)
            break

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact, aliases=['best'])
    wandb.finish()

if __name__ == '__main__':
    main()