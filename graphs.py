import matplotlib.pyplot as plt
import pandas as pd

seeds = [42, 43, 44, 45, 46]


def graph_f1():
    # CSVファイルのパスを指定
    files = {}
    for seed in seeds:
        file1 = f"result/tomato/{seed}/01_finetune/reports.csv"
        file2 = f"result/tomato/{seed}/02_dist_tune/reports.csv"
        file3 = f"result/tomato/{seed}/03_triplet_tune/reports.csv"
        files[seed] = [file1, file2, file3]

    # CSVファイルを読み込む
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()

    for seed in seeds:
        df1 = pd.concat([df1, pd.read_csv(files[seed][0], header=None, names=["Epoch", "Acc", "F1_Score"])])
        df2 = pd.concat([df2, pd.read_csv(files[seed][1], header=None, names=["Epoch", "Acc", "F1_Score"])])
        df3 = pd.concat([df3, pd.read_csv(files[seed][2], header=None, names=["Epoch", "Acc", "F1_Score"])])
    # エポック毎の平均と標準偏差を計算
    df1 = df1.groupby("Epoch").agg(["mean", "std"])
    df2 = df2.groupby("Epoch").agg(["mean", "std"])
    df3 = df3.groupby("Epoch").agg(["mean", "std"])

    print(df1)
    print(df2)
    print(df3)

    # F1-Scoreをエラーバー付きの散布図で描画
    plt.errorbar(df1.index - 20, df1["F1_Score"]["mean"], fmt="og", yerr=df1["F1_Score"]["std"], label="fine-tune", capsize=5)
    plt.errorbar(df2.index, df2["F1_Score"]["mean"], fmt="or", yerr=df2["F1_Score"]["std"], label="dist-tune", capsize=5)
    plt.errorbar(df3.index + 20, df3["F1_Score"]["mean"], fmt="ob", yerr=df3["F1_Score"]["std"], label="triplet-tune", capsize=5)

    # グラフにラベルを追加
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.legend()

    # 保存
    plt.savefig("result/report_graphs_f1.png")
    plt.show()


def graph_acc():
    # CSVファイルのパスを指定
    files = {}
    for seed in seeds:
        file1 = f"result/tomato/{seed}/01_finetune/reports.csv"
        file2 = f"result/tomato/{seed}/02_dist_tune/reports.csv"
        file3 = f"result/tomato/{seed}/03_triplet_tune/reports.csv"
        files[seed] = [file1, file2, file3]

    # CSVファイルを読み込む
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()

    for seed in seeds:
        df1 = pd.concat([df1, pd.read_csv(files[seed][0], header=None, names=["Epoch", "Acc", "F1_Score"])])
        df2 = pd.concat([df2, pd.read_csv(files[seed][1], header=None, names=["Epoch", "Acc", "F1_Score"])])
        df3 = pd.concat([df3, pd.read_csv(files[seed][2], header=None, names=["Epoch", "Acc", "F1_Score"])])
    # エポック毎の平均と標準偏差を計算
    df1 = df1.groupby("Epoch").agg(["mean", "std"])
    df2 = df2.groupby("Epoch").agg(["mean", "std"])
    df3 = df3.groupby("Epoch").agg(["mean", "std"])

    print(df1)
    print(df2)
    print(df3)

    # Accuracyをエラーバー付きの散布図で描画
    plt.errorbar(df1.index - 20, df1["Acc"]["mean"], fmt="og", yerr=df1["Acc"]["std"], label="fine-tune", capsize=5)
    plt.errorbar(df2.index, df2["Acc"]["mean"], fmt="or", yerr=df2["Acc"]["std"], label="dist-tune", capsize=5)
    plt.errorbar(df3.index + 20, df3["Acc"]["mean"], fmt="ob", yerr=df3["Acc"]["std"], label="triplet-tune", capsize=5)

    # グラフにラベルを追加
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # 保存
    plt.savefig("result/report_graphs_acc.png")
    plt.show()


if __name__ == "__main__":
    graph_acc()
