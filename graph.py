import matplotlib.pyplot as plt
import pandas as pd


def graph():
    # CSVファイルのパスを指定
    file1 = "result/cucumber/10leak/44/01_finetune/reports.csv"
    file2 = "result/cucumber/10leak/44/02_dist_tune/reports.csv"
    file3 = "result/cucumber/10leak/44/03_triplet_tune/reports.csv"
    file4 = "result/cucumber/10leak/44/04_dann_tune/reports.csv"
    file5 = "result/cucumber/10leak/44/05_dann/reports.csv"

    # CSVファイルを読み込む
    df1 = pd.read_csv(file1, header=None, names=["Epoch", "Acc1", "F1_Score1"])
    df2 = pd.read_csv(file2, header=None, names=["Epoch", "Acc2", "F1_Score2"])
    df3 = pd.read_csv(file3, header=None, names=["Epoch", "Acc3", "F1_Score3"])
    df4 = pd.read_csv(file4, header=None, names=["Epoch", "Acc4", "F1_Score4"])
    df5 = pd.read_csv(file5, header=None, names=["Epoch", "Acc5", "F1_Score5"])

    # F1-Scoreを折れ線グラフで描画
    plt.plot(df1["Epoch"], df1["Acc1"], label="fine-tune")
    plt.plot(df2["Epoch"], df2["Acc2"], label="dist-tune")
    plt.plot(df3["Epoch"], df3["Acc3"], label="triplet-tune")
    # plt.plot(df4["Epoch"], df4["Acc4"], label="dann-tune")
    plt.plot(df5["Epoch"], df5["Acc5"], label="dann")

    # グラフにラベルを追加
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend()

    # 範囲を指定
    # plt.xlim()
    # plt.ylim(80.0, 93.0)

    # グラフを表示
    plt.savefig("result/report_graph.png")
    plt.show()


if __name__ == "__main__":
    graph()
