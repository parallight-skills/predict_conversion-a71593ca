"""SKILL #1 —— 特征工程(「如何处理数据」)。
把原始 6 个特征变换成「更可学」的特征。

⚠️ 现在是身份变换(原样返回、啥也没做)——这就是为什么 raw 流水线只有约 0.55(≈瞎猜):
这道题的信号几乎全藏在**特征的交互**里(x_i·x_j)和**非线性**里(x_i²),纯线性模型抓不到。

指挥你的 agent 把它改成真特征工程:加上两两交互项 x_i*x_j、平方项 x_i**2,
让藏起来的非线性信号变得「线性可分」。这是本题**价值最高的一件 skill**。
约束:只看特征本身做变换,绝不读 ml/_truth/。"""

def engineer(X):
    # 原始特征 + 两两交互项 + 平方项
    # 6 原始 + 15 交互 + 6 平方 = 27
    result = []
    for row in X:
        feats = list(row)
        n = len(feats)
        # 交互项 x_i * x_j (i < j)
        for i in range(n):
            for j in range(i + 1, n):
                feats.append(feats[i] * feats[j])
        # 平方项 x_i^2
        for i in range(n):
            feats.append(feats[i] * feats[i])
        result.append(feats)
    return result
