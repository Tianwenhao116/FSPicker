import sys
import os
from os.path import dirname, abspath
import importlib
import json

# 动态导入相关模块
FSPickerHome = os.path.dirname(os.path.abspath(__file__))
FSPickerHome = os.path.split(FSPickerHome)[0]  # 获取项目根目录
sys.path.append(FSPickerHome)
sys.path.append(os.path.split(FSPickerHome)[0])

coords2labels = importlib.import_module(".utils.coords2labels", package=os.path.split(FSPickerHome)[1])
coord_gen = importlib.import_module(f".utils.coord_gen", package=os.path.split(FSPickerHome)[1])
norm = importlib.import_module(f".utils.normalization", package=os.path.split(FSPickerHome)[1])


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != '--pre_configs':
        raise ValueError("Missing required argument: --pre_configs <config_file>")
    pre_configs_path = sys.argv[2]

    with open(pre_configs_path, 'r') as f:
        pre_config = json.loads(f.read().lstrip('pre_config='))

    coord_gen.coords_gen_show(args=(pre_config["coord_path"],
                                    pre_config["coord_format"],
                                    pre_config["base_path"],
                                    None,
                                    ))

    # 归一化处理
    norm.norm_show(args=(pre_config["tomo_path"],
                         pre_config["tomo_format"],
                         pre_config["base_path"],
                         pre_config["norm_type"],
                         None,
                         ))

    # 根据坐标生成标签
    coords2labels.label_gen_show(args=(pre_config["base_path"],
                                       pre_config["coord_path"],
                                       pre_config["coord_format"],
                                       pre_config["tomo_path"],
                                       pre_config["tomo_format"],
                                       pre_config["num_cls"],
                                       'sphere',
                                       pre_config["label_diameter"],
                                       None,
                                       ))

    coords2labels.label_gen_show(args=(pre_config["base_path"],
                                       pre_config["coord_path"],
                                       pre_config["coord_format"],
                                       pre_config["tomo_path"],
                                       pre_config["tomo_format"],
                                       pre_config["num_cls"],
                                       'data_ocp',
                                       pre_config["ocp_diameter"],
                                       None,
                                       ))
