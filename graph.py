import matplotlib.pyplot as plt
import pandas as pd


def graph():
    # CSVファイルのパスを指定
    file1 = "result/tomato/44/01_fintune/reports.csv"
    file2 = "result/tomato/44/02_dist_tune/reports.csv"
    file3 = "result/tomato/44/03_triplet_tune/reports.csv"

    # CSVファイルを読み込む
    df1 = pd.read_csv(file1, header=None, names=["Epoch", "Acc1", "F1_Score1"])
    df2 = pd.read_csv(file2, header=None, names=["Epoch", "Acc2", "F1_Score2"])
    df3 = pd.read_csv(file3, header=None, names=["Epoch", "Acc3", "F1_Score3"])

    # F1-Scoreを折れ線グラフで描画
    plt.plot(df1["Epoch"], df1["F1_Score1"], label="fine-tune")
    plt.plot(df2["Epoch"], df2["F1_Score2"], label="dist-tune")
    plt.plot(df3["Epoch"], df3["F1_Score3"], label="triplet-tune")

    # グラフにラベルを追加
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.legend()

    # 範囲を指定
    plt.xlim(100, 1000)
    plt.ylim(60.0, 80.0)

    # グラフを表示
    plt.savefig("result/report_graph.png")
    plt.show()


if __name__ == "__main__":
    graph()
