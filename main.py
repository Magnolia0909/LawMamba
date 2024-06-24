import torch
import argparse
import numpy as np

from text_entailment.te_extract_law import extract_law
from text_entailment.te_dataloader import TeDataLoader
from text_entailment.te_trainer import Te_Trainer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# from check.ch_dataload import ChDataLoader
# from check.ch_dataset import ChDataset
# from check.ch_trainer import Ch_Trainer
#
# from text_entailment.te_tokenizer import Tokenizer
# from dp.dp_dataloader import DPDataLoader
# from dp.metrics import compute_scores
# from dp.optimizers import build_optimizer, build_lr_scheduler
# from dp.dp_trainer import Trainer
# from dp.loss import compute_loss
# from dp.dp_generate import DPGenerateModel


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')

def parse_agrs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/home/sxx/experiment/data/law_data/law.jsonl', help='判定要素一致性的数据集')
    parser.add_argument('--stopword_path', type=str, default='/home/sxx/experiment/data/law_data/stopword.txt', help='停用词')
    parser.add_argument('--output_dir', type=str, default='result/', help='存放要素一致性判定结果')

    parser.add_argument('--check_law_file_path', type=str, default='/home/sxx/experiment/data/law_data/civil.json', help='存放法律条文的数据集')
    parser.add_argument('--check_contract_file_path', type=str, default='/homte_0608.loge/sxx/experiment/data/law_data/contract.json', help='存放合同条款的数据集')
    parser.add_argument('--num_labels', type=str, default=16, help='法律分类数量')
    parser.add_argument('--model_choice', type=str, default='lstm', choices=['kan', 'transformer', 'rnn', 'bert', 'mamba', 'gpt'], help='使用的模型')
    parser.add_argument('--is_multi', type=bool, default=False, help='是否启动多任务')
    parser.add_argument('--multi_tasks', type=int, default=2, help='多任务个数')
    parser.add_argument('--multi_begin', type=int, default=10, help='多任务启动轮次')
    parser.add_argument('--residual_weight', default=0.8, type=float, help='the weight of residual operation for pooling')

    parser.add_argument('--te_train_data', type=str, default='/home/sxx/experiment/data/law_data/train_json_v3.json', help='要素一致性排名训练集')
    parser.add_argument('--te_test_data', type=str, default='/home/sxx/experiment/data/law_data/test_json_v3.json', help='要素一致性排名测试集')
    parser.add_argument('--te_vocab_file', type=str, default='/home/sxx/experiment/data/law_data/te/vocab.txt', help='要素一致性排名测试集')
    parser.add_argument('--law_data', type=str, default='/home/sxx/experiment/data/law_data/law.csv', help='存放法律的数据集')

    parser.add_argument('--t_d_model', type=int, default=384, help='Dimension of the model')
    parser.add_argument('--t_hidden', type=int, default=256, help='Dimension of inner layers')
    parser.add_argument('--t_num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--t_n_head', type=int, default=8, help='Number of head')

    parser.add_argument('--kan_n_embd', type=int, default=512, help='Number of head')
    parser.add_argument('--kan_n_head', type=int, default=8, help='Number of head')
    parser.add_argument('--kan_attn_pdrop', type=int, default=0.1, help='Number of head')
    parser.add_argument('--kan_resid_pdrop', type=int, default=0.1, help='Number of head')
    parser.add_argument('--kan_block_size', type=int, default=128, help='Number of head')

    parser.add_argument('--clip_context_length', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--clip_transformer_width', type=int, default=512, help='Dimension of inner layers')
    parser.add_argument('--clip_transformer_layers', type=int, default=6, help='clip layers')
    parser.add_argument('--clip_transformer_heads', type=int, default=8, help='clip head number')
    parser.add_argument('--clip_embed_dim', type=int, default=512, help='clip embed dim')

    parser.add_argument("--private", type=str2bool, nargs='?', const=True, default=False, help='If privatization should be applied during pre-training')
    parser.add_argument("--no_clipping", type=str2bool, nargs='?', const=True, default=False, help='Whether or not to clip encoder hidden states in the non-private setting.')
    parser.add_argument("--epsilon", type=float, default=1.0, help='value of epsilon')


    parser.add_argument('--gpt2_dim', type=int, default=768, help='gpt2')
    parser.add_argument('--example_num', type=int, default=3, help='一条合同对应的法律数量')
    parser.add_argument('--grad_clip', default=0.1, type=float, help='Gradient clipping threshold.')
    parser.add_argument('--cross_attn', default="contract", help='law|contract')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=128, type=int, help='输入合同的维度')
    parser.add_argument('--embed_size', default=384, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--hidden_size', default=256, type=int, help='文本分类任务隐含向量维度')
    parser.add_argument('--precomp_enc_type', default="basic", help='basic|weight_norm')
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--word_dim', default=256, type=int, help='Dimensionality of the word embedding.')
    # parser.add_argument('--num_layers', default=1, type=int, help='Number of GRU layers.')
    parser.add_argument('--bi_gru', default=True, action='store_true', help='Use bidirectional GRU.')
    parser.add_argument('--no_txtnorm', action='store_true', help='Do not normalize the text embeddings.')
    parser.add_argument('--vocab_size', default=4354, type=int, help='词汇表长度')
    parser.add_argument('--lambda_softmax', default=9., type=float, help='Attention softmax temperature.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm", help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func' , default="Mean", help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--lambda_lse', default=6., type=float, help='LogSumExp temp.')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size 的值')
    parser.add_argument('--num_workers', type=int, default=0, help='并行工作数')
    parser.add_argument('--max_seq_length', type=int, default=128, help='句子最大长度')
    parser.add_argument('--n_gpu', type=int, default=1, help='使用的gpu的数量')
    parser.add_argument('--lr_te', type=float, default=5e-5, help='文本蕴含任务的学习率')
    parser.add_argument('--epochs', type=int, default=100, help='文本蕴含任务的epoch数量')
    parser.add_argument('--save_period', type=int, default=1, help='模型保存周期')
    parser.add_argument('--save_te_dir', type=str, default='result/te', help='文本蕴含任务保存地址')
    parser.add_argument('--save_dp_dir', type=str, default='result/dp', help='差分隐私重写器保存地址')
    parser.add_argument('--record_te_dir', type=str, default='records/te/')
    parser.add_argument('--record_dp_dir', type=str, default='records/dp/')
    parser.add_argument('--log_dir', type=str, default='./records/log/te/')
    parser.add_argument('--csv_dir', type=str, default='./records/log/te/aux_weight.log')
    parser.add_argument('--log_dp_dir', type=str, default='./records/log/dp/')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')
    parser.add_argument('--resume', type=str, default=None,  help='whether to resume the training from existing checkpoints.')
    # parser.add_argument('--resume', type=str, default="/home/sxx/experiment/law/law_consistence/result/dp/current_checkpoint.pth",  help='whether to resume the training from existing checkpoints.')

    parser.add_argument('--seed', type=int, default=1111, help='固定随机种子')
    # parser.add_argument('--seed', type=int, default=42, help='固定随机种子')

    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-5, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=512, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=2, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=3, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    args = parser.parse_args()
    return args

def main():
    args = parse_agrs()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    law_data = extract_law(args)
    '''
    差分隐私重写器
    '''
    # dp_tokenizer = Tokenizer(args)
    # dp_train_dataloader = DPDataLoader(args, dp_tokenizer, split='train', shuffle=True)
    # dp_test_dataloader = DPDataLoader(args, dp_tokenizer, split='test', shuffle=False)
    #
    # dp_model = DPGenerateModel(args, dp_tokenizer)
    # dp_criterion = compute_loss
    # dp_metrics = compute_scores
    # dp_optimizer = build_optimizer(args, dp_model)
    # dp_lr_scheduler = build_lr_scheduler(args, dp_optimizer)
    #
    # # build trainer and start to train
    # dp_trainer = Trainer(dp_model, dp_criterion, dp_metrics, dp_optimizer, args, dp_lr_scheduler, dp_train_dataloader, dp_test_dataloader)
    # dp_trainer.train()

    '''
    文本蕴含任务训练器
    '''
    te_train_loader = TeDataLoader(args, law_data, split='train', shuffle=True)
    te_test_loader = TeDataLoader(args, law_data, split='test', shuffle=False)
    te_trainer = Te_Trainer(args, te_train_loader, te_test_loader, law_data)
    te_trainer.train()

    '''
    一致性校验训练器
    '''
    # ch_train_loader = ChDataLoader(args, tokenizer, split="train", shuffle=True)
    # ch_test_loader = ChDataLoader(args, tokenizer, split="test", shuffle=True)
    # ch_trainer = Ch_Trainer(args, tokenizer, ch_train_loader, ch_test_loader)
    # ch_trainer.train()

if __name__ == "__main__":
    main()