import torch.nn as nn

import datasets
import models

import datetime

from prettytable import PrettyTable

from models import TransformerNLI
from utils import *


class Evaluate():
    def __init__(self):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.args = parse_args()
        self.device = get_device(self.args.gpu)
        self.logger = get_logger(self.args, "evaluate")

        self.dataset_options = {
            'batch_size': self.args.batch_size,
        }
        self.dataset = datasets.__dict__[self.args.dataset](self.dataset_options)

        self.validation_accuracy, model_dict = self.load_model()
        self.model = TransformerNLI(self.args, self.dataset.vocab, self.dataset.label_vocab)
        self.model.to(self.device)
        self.model.load_state_dict(model_dict)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.test_accuracy = -1
        print("resource preparation done: {}".format(datetime.datetime.now()))

    def load_model(self):
        model = torch.load(
            '{}/{}/best-{}-params.pt'.format(self.args.results_dir, self.args.dataset,
                                             self.args.dataset))
        return model['accuracy'], model['model_dict']

    def print_confusion_matrix(self, labels, confusion_matrix):
        table = PrettyTable()
        table.field_names = ["confusion-matrix"] + ["{}-pred".format(label) for label in labels] + ["total"]
        row_label = ["{}-actual".format(label) for label in labels] + ["total"]

        row_sum = confusion_matrix.sum(dim=1).view(len(labels), 1)
        confusion_matrix = torch.cat((confusion_matrix, row_sum), dim=1)
        col_sum = confusion_matrix.sum(dim=0).view(1, len(labels) + 1)
        confusion_matrix = torch.cat((confusion_matrix, col_sum), dim=0)
        confusion_matrix = confusion_matrix.tolist()
        for i, row in enumerate(confusion_matrix):
            table.add_row([row_label[i]] + [int(count) for count in row])
        print(table)
        self.logger.info(table)

    def label_wise_accuracy(self, lable_map, confusion_matrix):
        table = PrettyTable()
        table.field_names = ["label", "accuracy"]
        for label, value in lable_map.items():
            acc = round((100. * confusion_matrix[value][value] / confusion_matrix[value].sum()).item(), 3)
            table.add_row([label, acc])
        print(table)
        self.logger.info(table)

    def evaluate(self):
        self.model.eval()
        n_correct, n_total, n_loss = 0, 0, 0
        labels = self.dataset.labels().copy()
        confusion_matrix = torch.zeros(len(labels), len(labels))

        with torch.no_grad():
            for x, y in self.dataset.test_iter:
                premise = x["premise"].cuda(self.device)
                hypothesis = x["hypothesis"].cuda(self.device)
                label = y["label"].cuda(self.device)
                answer = self.model(premise, hypothesis)
                loss = self.criterion(answer, label)

                n_correct += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
                n_total += len(label)
                n_loss += loss.item()

            test_loss = n_loss / n_total
            test_acc = 100. * n_correct / n_total
            return test_loss, test_acc, confusion_matrix

    def execute(self):
        _, test_acc, confusion_matrix = self.evaluate()
        table = PrettyTable()
        table.field_names = ["data", "accuracy"]
        table.add_row(["validation", round(self.validation_accuracy, 3)])
        table.add_row(["test", round(test_acc, 3)])
        print(table)
        self.logger.info(table)
        lable_map = self.dataset.labels()
        self.label_wise_accuracy(lable_map, confusion_matrix)
        self.print_confusion_matrix(lable_map.keys(), confusion_matrix)


task = Evaluate()
task.execute()
