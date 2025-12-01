import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)
epochs = np.arange(1, 101)


def simulate_loss(L_max, L_min, decay_rate=0.15, noise_scale=0.003):
    """生成平滑衰减型 loss 曲线"""
    loss = L_min + (L_max - L_min) * np.exp(-decay_rate * (epochs - 1))
    # 添加微小波动
    noise = np.random.normal(0, noise_scale * (L_max - L_min), size=len(epochs))
    loss += noise
    return loss


# 模拟4条不同初始值的曲线
loss_curves = [
    simulate_loss(10, 5, decay_rate=0.09),
]

# 保存到CSV
df = pd.DataFrame({"Epoch": epochs})
for i, loss in enumerate(loss_curves, start=1):
    df[f"Loss_Model_{i}"] = loss
csv_path = "simulated_realistic_loss_curves.csv"
df.to_csv(csv_path, index=False)
print(f"✅ 模拟 loss 数据已保存到: {csv_path}")

# 绘图（仿造示例）
plt.figure(figsize=(12, 3))
for i, loss in enumerate(loss_curves, start=1):
    plt.subplot(1, 4, i)
    plt.plot(epochs, loss, color="red", linewidth=1.2)
    plt.xlabel("Epoch", fontsize=9)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
plt.savefig("simulated_realistic_loss_curves.png", dpi=300)
plt.show()
