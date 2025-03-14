import json
import sys
import os
from os.path import dirname, abspath
import importlib
import numpy as np

FSPickerHome = dirname(abspath(__file__))
FSPickerHome = os.path.split(FSPickerHome)[0]
sys.path.append(FSPickerHome)
sys.path.append(os.path.split(FSPickerHome)[0])
train = importlib.import_module(".train", package=os.path.split(FSPickerHome)[1])
option = importlib.import_module(f".options.option", package=os.path.split(FSPickerHome)[1])


if __name__ == '__main__':
    options = option.BaseOptions()
    args = options.gather_options()

    # cofig只有cfg里的参数是用户指定的其余是默认参数
    with open(args.train_configs, 'r') as f:
        cfg = json.loads(''.join(f.readlines()).lstrip('train_configs='))

    # parameters默认使用了
    args.use_bg = True
    args.use_IP = True
    args.test_use_pad = True
    args.meanPool_NMS = True
    args.f_maps = [24, 48, 72, 108]
    args.num_classes = cfg['num_cls']
    train_cls_num = cfg['num_cls']
    if args.num_classes == 1:
        args.use_sigmoid = True
        args.use_softmax = False
    else:
        train_cls_num = train_cls_num + 1
        args.use_sigmoid = False
        args.use_softmax = True
    args.batch_size = cfg['batch_size']
    args.block_size = cfg['patch_size']
    args.val_batch_size = args.batch_size
    args.val_block_size = args.block_size
    args.pad_size = [cfg['padding_size']]
    args.learning_rate = cfg['lr']
    args.max_epoch = cfg['max_epochs']
    args.threshold = cfg['seg_thresh']
    args.gpu_id = [int(i) for i in cfg['gpu_ids'].split(',')]
    args.configs = args.train_configs
    args.test_mode = 'val'
    args.train_set_ids = cfg['train_set_ids']#训练集编号
    args.val_set_ids = cfg['val_set_ids']#测试集编号
    args.num_classes = train_cls_num
    #解析参数获取训练集和验证集id

    train_list = []
    for item in args.train_set_ids.split(','):
        if '-' in item:
            tmp = [int(i) for i in item.split('-')]
            train_list.extend(np.arange(tmp[0], tmp[1] + 1).tolist())
        else:
            train_list.append(int(item))

    val_list = []
    for item in args.val_set_ids.split(','):
        if '-' in item:
            tmp = [int(i) for i in item.split('-')]
            val_list.extend(np.arange(tmp[0], tmp[1] + 1).tolist())
        else:
            val_list.append(int(item))
    val_first = len(train_list) if val_list[0] not in train_list else len(train_list) - 1
    args.data_split = [0, len(train_list),  # train
                       val_first, val_first + 1,  # val
                       val_first, val_first + 1]  # test_val

    for k, v in sorted(vars(args).items()):
        print(k, '=', v)

    # Training
    train.train_func(args, stdout=None)