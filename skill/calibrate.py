"""SKILL #3 —— 校准 / 预测(「如何 predict」)。
把模型输出的概率变成最终的 0/1 标签。

⚠️ 现在是固定 0.5 阈值。指挥 agent 改成**在训练集上挑一个让训练 accuracy 最高的阈值**
(threshold tuning)——类别不均衡时,这一步能再抠几分。这是价值最小、但也最便宜的一件 skill。"""


def to_labels(train_probs, train_y, test_probs):
    # 在训练集上搜最优阈值:试 0.01 到 0.99 共 99 个候选,挑训练 accuracy 最高那个
    best_thr, best_acc = 0.5, 0.0
    for thr in [i / 100.0 for i in range(1, 100)]:
        preds = [1 if p >= thr else 0 for p in train_probs]
        acc = sum(1 for p, t in zip(preds, train_y) if int(p) == int(t)) / len(train_y)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return [1 if p >= best_thr else 0 for p in test_probs]
