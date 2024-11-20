# dataset
# src_path = "/data2/eto/Dataset/eggplant_fewclass/7class_leak/train/"
# tgt_path = "/data2/eto/Dataset/eggplant_fewclass/7class_leak/fewshot/"
# test_path = "/data2/eto/Dataset/eggplant_fewclass/7class_leak/test/"
src_path = "/data2/eto/Dataset/tomato/train/"
tgt_path = "/data2/eto/Dataset/tomato/fewshot/"
test_path = "/data2/eto/Dataset/tomato/test/"

pretrain_batch_size = 128
batch_size = 12 * 2

# learning parameters
seed = 42
lr = 1e-3
pretrain_num_epochs = 100
finetune_num_epochs = 1000

# model weight
pretrain_encoder_weight = "./weight/encoder.pth"
pretrain_classifier_weight = "./weight/classifier.pth"
# dann_model_weight = "./weight/dann.pth"
dann_model_weight = "./weight/tomato/dann.pth"
finetune_encoder_weight = "./weight/finetune_encoder.pth"
finetune_classifier_weight = "./weight/finetune_classifier.pth"
dist_tune_encoder_weight = "./weight/dist_tune_encoder.pth"
dist_tune_classifier_weight = "./weight/dist_tune_classifier.pth"
triplet_tune_encoder_weight = "./weight/triplet_tune_encoder.pth"
triplet_tune_classifier_weight = "./weight/triplet_tune_classifier.pth"
cosine_tune_encoder_weight = "./weight/cosine_tune_encoder.pth"
cosine_tune_classifier_weight = "./weight/cosine_tune_classifier.pth"
dann_tune_model_weight = "./weight/dann_tune.pth"

# process setting
gpu_ids = "6,7"
num_workers = 16

test_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
