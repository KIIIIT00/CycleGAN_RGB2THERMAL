import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')


# lossのテキストファイル
loss_txt = "./checkpoints/Scene2ver2_400/loss_log.txt"

# テキストファイルの読み込み
f = open(loss_txt, 'r')
datalist = f.readlines()
f.close()

# 2行目を辞書型に変換
dict_txt = datalist[1].replace(",", "").replace(": ", "\":").replace(" ", ", \"").replace(")", "").replace("(", "{\"").replace(", \"\n", "}")
loss_dict = eval(dict_txt) # 文字列でのコード実行
# データフレーム型に変換
df_loss = pd.DataFrame(loss_dict,index=[0,])

# 2行目以降の処理
for i in range(2, len(datalist)):
    # 2行目以降を辞書型に変換
    dict_txt = datalist[i].replace(",", "").replace(": ", "\":").replace(" ", ", \"").replace(")", "").replace("(", "{\"").replace(", \"\n", "}")
    loss_dict = eval(dict_txt) # 文字列でのコード実行
    # データフレーム型に変換
    df_dict = pd.DataFrame(loss_dict,index=[i-1,])
    # df_lossに結合
    df_loss = pd.concat([df_loss, df_dict], axis=0)

# indexを追加
df_loss = df_loss.reset_index()

print(df_loss.head())
# 4行2列のグラフを作成
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(18,24))

# Adversarial Loss(D)
sns.lineplot(x='index', y='D_A', data=df_loss, ax=ax1).set_title('loss_D_A')
sns.lineplot(x='index', y='D_B', data=df_loss, ax=ax2).set_title('loss_D_B')

# Adversarial Loss(G)
sns.lineplot(x='index', y='G_A', data=df_loss, ax=ax3).set_title('loss_G_A')
sns.lineplot(x='index', y='G_B', data=df_loss, ax=ax4).set_title('loss_G_B')
# Cycle Consistency Loss
sns.lineplot(x='index', y='cycle_A', data=df_loss, ax=ax5).set_title('loss_cycle_A')
sns.lineplot(x='index', y='cycle_B', data=df_loss, ax=ax6).set_title('loss_cycle_B')
# Identity Mapping Loss
sns.lineplot(x='index', y='idt_A', data=df_loss, ax=ax7).set_title('loss_idt_A')
sns.lineplot(x='index', y='idt_B', data=df_loss, ax=ax8).set_title('loss_idt_B')
plt.figure(figsize=(10, 5))
plt.plot(df_loss.index, df_loss['D_A'], label = 'LossD_A')
plt.plot(df_loss.index, df_loss['G_A'], label = 'LossG_A')
plt.title('LossD_A and LossG_A')
plt.ylabel('LossA')
plt.xlabel('data')
plt.legend()
plt.show()