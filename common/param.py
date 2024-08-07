# dataset
src_path = "/data2/eto/Dataset/tomato/train/"
tgt_path = "/data2/eto/Dataset/tomato/fewshot/"
test_path = "/data2/eto/Dataset/tomato/test/"
pretrain_batch_size = 128
batch_size = 12 * 3

# learning parameters
seed = 55
lr = 1e-3
pretrain_num_epochs = 100
finetune_num_epochs = 1000

# model weight
pretrain_encoder_weight = "./weight/encoder.pth"
pretrain_classifier_weight = "./weight/classifier.pth"
finetune_encoder_weight = "./weight/finetune_encoder.pth"
finetune_classifier_weight = "./weight/finetune_classifier.pth"
dist_tune_encoder_weight = "./weight/dist_tune_encoder.pth"
dist_tune_classifier_weight = "./weight/dist_tune_classifier.pth"
triplet_tune_encoder_weight = "./weight/triplet_tune_encoder.pth"
triplet_tune_classifier_weight = "./weight/triplet_tune_classifier.pth"
cosine_tune_encoder_weight = "./weight/cosine_tune_encoder.pth"
cosine_tune_classifier_weight = "./weight/cosine_tune_classifier.pth"
dann_tune_model_weight = "./weight/dann_tune.pth"

# GPU settings
gpu_ids = "4,5"
