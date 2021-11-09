import time
import datetime
from torch import optim
import datasets
from labelSmoothedCrossEntropy import LabelSmoothedCrossEntropyCriterion
from models import TransformerNLI
from opt import InverseSquareRootScheduleOpt

from utils import *

import global_value as gol

gol._init()


class Train():
    def __init__(self):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.args = parse_args()
        self.device = get_device(self.args.gpu)
        self.logger = get_logger(self.args, "train")
        self.logger.info("Arguments: {}".format(self.args))

        dataset_options = {
            'batch_size': self.args.batch_size,
        }
        self.dataset = datasets.__dict__[self.args.dataset](dataset_options)

        self.model = TransformerNLI(self.args, self.dataset.vocab, self.dataset.label_vocab)

        self.model.to(self.device)
        self.criterion = LabelSmoothedCrossEntropyCriterion(eps=0.1, reduce=True)
        self.opt = InverseSquareRootScheduleOpt(optim.Adam(self.model.parameters(), lr=self.args.lr,
                                                           weight_decay=self.args.weight_decay),
                                                self.args.lr, warmup_updates=self.args.warmup_updates)
        self.best_val_acc = None

        print("resource preparation done: {}".format(datetime.datetime.now()))

    def result_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, took):
        if self.best_val_acc is None or val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save({
                'accuracy': self.best_val_acc,
                'model_dict': self.model.state_dict(),
            }, '{}/{}/{}/best-params.pt'.format(self.args.results_dir, self.args.dataset, self.args.model))
        self.logger.info(
            '| Epoch {:3d} | train loss {:5.3f} | train acc {:5.3f} '
            '| val loss {:5.3f} | val acc {:5.3f} | time: {:5.3f}s |'
            .format(epoch, train_loss, train_acc, val_loss, val_acc, took))

    def train(self):
        self.model.train()
        n_correct, n_total, n_loss = 0, 0, 0
        for x, y in self.dataset.train_iter:
            premise = x["premise"].cuda(self.device)
            hypothesis = x["hypothesis"].cuda(self.device)
            label = y["label"].cuda(self.device)

            self.opt.zero_grad()
            answer = self.model(premise, hypothesis)
            loss = self.criterion(answer, label)

            n_correct += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
            n_total += len(label)
            n_loss += loss.item()
            loss.backward()
            self.opt.step()
        train_loss = n_loss / n_total
        train_acc = 100. * n_correct / n_total
        return train_loss, train_acc

    def validate(self):
        self.model.eval()
        n_correct, n_total, n_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in self.dataset.dev_iter:
                premise = x["premise"].cuda(self.device)
                hypothesis = x["hypothesis"].cuda(self.device)
                label = y["label"].cuda(self.device)
                answer = self.model(premise, hypothesis)
                loss = self.criterion(answer, label)

                n_correct += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
                n_total += len(label)
                n_loss += loss.item()

            val_loss = n_loss / n_total
            val_acc = 100. * n_correct / n_total
            return val_loss, val_acc

    def execute(self):
        print(" [*] Training starts!")
        print('-' * 99)
        for epoch in range(1, self.args.epochs + 1):
            gol.set_value("epoch", epoch)
            start = time.time()

            train_loss, train_acc = self.train()
            val_loss, val_acc = self.validate()

            took = time.time() - start
            self.result_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, took)

            print('| Epoch {:3d} | train loss {:5.3f} | train acc {:5.3f} '
                  '| val loss {:5.3f} | val acc {:5.3f} | lr {:1.7f} | time: {:5.3f}s |'.
                  format(epoch, train_loss, train_acc, val_loss, val_acc, self.opt.get_lr(), took))
        self.finish()

    def finish(self):
        self.logger.info("[*] Training finished!\n\n")
        print('-' * 120)
        print(" [*] Training finished!")
        print(" [*] Please find the saved model and training log in results_dir")


task = Train()
task.execute()
