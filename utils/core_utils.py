import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.utils import *
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import f1_score

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_all = {'y_true': [], 'y_pred': []}

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        if count == 0:
            acc = None
        else:
            acc = float(correct) / count
        return acc, correct, count

    def get_f1(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """Train for a single fold"""
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    if args.model_size:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.n_classes)
    model = model.to(device)
    print('Done!')
    args.weight_decay = 0.0001

    print('\nInit optimizer ...', end=' ')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True) if args.early_stopping else None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    _, val_error, val_auc, val_logger = summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    val_f1 = val_logger.get_f1()
    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    test_f1 = acc_logger.get_f1()
    if writer:
        writer.add_scalar('final/val_f1', val_f1, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, test_f1, val_f1

def test(datasets, cur, args):
    """Train for a single fold"""
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    if args.model_size:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.n_classes)
    model = model.to(device)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')

    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    test_f1 = acc_logger.get_f1()
    if writer:
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, test_f1

def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    train_error = 0.
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits = model(data)
        loss = loss_fn(logits, label)
        preds = logits.argmax(dim=1)

        acc_logger.log_batch(preds, label)
        error = calculate_error(preds, label)

        train_error += error
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(loader)
    train_error /= len(loader)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))

def validate(cur, epoch, model, loader, n_classes, early_stopping, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    val_error = 0.
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    results_dict = {}
    probs = np.zeros((len(loader.dataset), n_classes))
    labels = np.zeros(len(loader.dataset))
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits = model(data)
            loss = loss_fn(logits, label)
            preds = logits.argmax(dim=1)

            probs[batch_idx * loader.batch_size:batch_idx * loader.batch_size + len(label)] = torch.softmax(logits, dim=1).detach().cpu().numpy()
            labels[batch_idx * loader.batch_size:batch_idx * loader.batch_size + len(label)] = label.cpu().numpy()

            acc_logger.log_batch(preds, label)
            error = calculate_error(preds, label)

            val_error += error
            val_loss += loss.item()
            
    val_loss /= len(loader)
    val_error /= len(loader)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('Epoch: {}, val_loss: {:.4f}, val_error: {:.4f}'.format(epoch, val_loss, val_error))

    val_auc = 0
    if n_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
        val_auc = auc
        print('Epoch: {}, val_auc: {:.4f}'.format(epoch, auc))
        if writer:
            writer.add_scalar('val/auc', auc, epoch)
        if early_stopping:
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

    return early_stopping.early_stop if early_stopping else False

def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    results_dict = {}
    probs = np.zeros((len(loader.dataset), n_classes))
    labels = np.zeros(len(loader.dataset))
    all_logits = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            all_logits.extend(logits.cpu().numpy())

            probs[batch_idx * loader.batch_size:batch_idx * loader.batch_size + len(label)] = torch.softmax(logits, dim=1).detach().cpu().numpy()
            labels[batch_idx * loader.batch_size:batch_idx * loader.batch_size + len(label)] = label.cpu().numpy()

            acc_logger.log_batch(preds, label)
            error = calculate_error(preds, label)

    test_error = calculate_error(np.argmax(probs, axis=1), labels)
    if n_classes == 2:
        test_auc = roc_auc_score(labels, probs[:, 1])
    else:
        fpr = {}
        tpr = {}
        roc_auc = {}
        labels_onehot = label_binarize(labels, classes=[i for i in range(n_classes)])
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], probs[:, i])
            roc_auc[i] = calc_auc(fpr[i], tpr[i])
        test_auc = np.mean([v for k, v in roc_auc.items()])

    return results_dict, test_error, test_auc, acc_logger
