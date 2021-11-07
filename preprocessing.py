from fastNLP import DataSet
from fastNLP import Vocabulary
from tqdm import tqdm
import spacy

train_file = ".data/snli/snli_1.0/snli_1.0_train.txt"
dev_file = ".data/snli/snli_1.0/snli_1.0_dev.txt"
test_file = ".data/snli/snli_1.0/snli_1.0_test.txt"

train_file_target = ".data/snli/train.txt"
dev_file_target = ".data/snli/dev.txt"
test_file_target = ".data/snli/test.txt"


def preData(path, target_path, lower=False):
    tokenizer = spacy.load("en_core_web_sm")
    sentence1 = []
    sentence2 = []
    gold_label = []
    with open(path, "r") as f:
        lines = f.readlines()
        headers = lines[0][:-1].split("\t")
        for line in lines[1:]:
            for header, content in zip(headers, line[:-1].split("\t")):
                if header == "sentence1":
                    sent = content.lower() if lower else content
                    sentence1.append(sent)
                elif header == "sentence2":
                    sent = content.lower() if lower else content
                    sentence2.append(sent)
                elif header in "gold_label":
                    gold_label.append(content.lower() if lower else content)

    data = {
        "premise": [],
        "hypothesis": [],
        "label": []
    }

    for sent in tqdm(tokenizer.pipe(sentence1, n_threads=64)):
        data["premise"].append(" ".join([token.orth_ for token in sent]))
    for sent in tqdm(tokenizer.pipe(sentence2, n_threads=64)):
        data["hypothesis"].append(" ".join([token.orth_ for token in sent]))
    data["label"] = gold_label

    total = 0
    cur = 0
    with open(target_path, "w") as f:
        f.writelines("\t".join(data.keys()) + "\n")
        for s1, s2, t in zip(data["premise"], data["hypothesis"], data["label"]):
            total += 1
            if t in ["neutral", "entailment", "contradiction"]:
                cur += 1
                f.writelines(s1 + "\t" + s2 + "\t" + t + "\n")
    print("%s(%d) -> %s(%d)" % (path, total, target_path, cur))
    return data


def preSNLI():
    preData(train_file, train_file_target)
    preData(dev_file, dev_file_target)
    preData(test_file, test_file_target)


if __name__ == "__main__":
    preSNLI()
