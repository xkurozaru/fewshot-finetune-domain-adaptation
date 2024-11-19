import os
import os.path as osp
import random
import shutil

from common import param, set_seed

N = 10
TEST_PATH = param.test_path
TGT_PATH = param.tgt_path


def leaking():
    set_seed(param.seed)

    # クラス名を取得
    classes = os.listdir(param.test_path)
    classes.sort()

    # テストデータから各クラスN枚、ランダムに抽出
    for cls in classes:
        img_paths = os.listdir(osp.join(param.test_path, cls))
        img_paths.sort()
        img_paths = random.sample(img_paths, N)

        # 各クラスの画像をfew_shotディレクトリに移動
        for img_path in img_paths:
            if not osp.exists(osp.join(param.tgt_path, cls)):
                os.makedirs(osp.join(param.tgt_path, cls))

            shutil.move(
                osp.join(param.test_path, cls, img_path),
                osp.join(param.tgt_path, cls, img_path),
            )
            print(img_path + " is leaked.")


def restoreing():
    # クラス名を取得
    classes = os.listdir(param.tgt_path)
    classes.sort()

    # few_shotディレクトリからすべての画像をテストデータに移動
    for cls in classes:
        img_paths = os.listdir(osp.join(param.tgt_path, cls))
        img_paths.sort()

        # 各クラスの画像をfew_shotディレクトリに移動
        for img_path in img_paths:
            shutil.move(
                osp.join(param.tgt_path, cls, img_path),
                osp.join(param.test_path, cls, img_path),
            )
            print(img_path + " is restored.")


if __name__ == "__main__":
    restoreing()
    # leaking()
