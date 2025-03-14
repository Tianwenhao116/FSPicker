import mrcfile
from multiprocessing import Pool
import pandas as pd
import os
import numpy as np
from glob import glob
import sys
import traceback

def gaussian3D(shape, sigma=1):
    l, m, n = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-l:l + 1, -m:m + 1, -n:n + 1]
    sigma = (sigma - 1.) / 2.
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    # h[h < np.finfo(float).eps * h.max()] = 0
    return h

def gaussian3D_weighted(shape, sigma=1):
    l, m, n = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-l:l + 1, -m:m + 1, -n:n + 1]
    sigma = (sigma - 1.) / 2.
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    boundary_weight = 1 + (np.sqrt(x**2 + y**2 + z**2) / (max(l, m, n)))
    return h * boundary_weight
class Coord_to_Label():
    def __init__(self, base_path, coord_path, coord_format, tomo_path, tomo_format,
                 num_cls, label_type, label_diameter):

        self.base_path = base_path
        self.coord_path = coord_path
        self.coord_format = coord_format
        self.tomo_path = tomo_path
        self.tomo_format = tomo_format
        self.num_cls = num_cls#粒子种类
        self.label_type = label_type
        if not isinstance(label_diameter, int):
            self.label_diameter = [int(i) for i in label_diameter.split(',')]
        else:
            self.label_diameter = [label_diameter]

        if 'ocp' in self.label_type.lower():
            self.label_path = os.path.join(self.base_path, self.label_type)
        else:
            self.label_path = os.path.join(self.base_path,
                                           self.label_type + str(self.label_diameter[0]))
        os.makedirs(self.label_path, exist_ok=True)

        self.dir_list = [i[:-len(self.coord_format)] for i in os.listdir(self.coord_path) if self.coord_format in i]
        self.names = [i + self.tomo_format for i in self.dir_list]

    def single_handle(self, i):
        self.tomo_file = f"{self.tomo_path}/{self.names[i]}"
        data_file = mrcfile.open(self.tomo_file, permissive=True)
        # print(os.path.join(self.label_path, self.names[i]))
        label_file = mrcfile.new(os.path.join(self.label_path, self.names[i]),
                                 overwrite=True)#新建一个文件存放标签

        label_positions = pd.read_csv(os.path.join(self.base_path, 'coords', '%s.coords' % self.dir_list[i]), sep='\t',
                                      header=None).to_numpy()

        # template = np.fromfunction(lambda i, j, k: (i - r) * (i - r) + (j - r) * (j - r) + (k - r) * (k - r) <= r * r,
        #                            (2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=int).astype(int)

        z_max, y_max, x_max = data_file.data.shape
        try:
            label_data = np.zeros(data_file.data.shape, dtype=np.float)
        except:
            label_data = np.zeros(data_file.data.shape, dtype=np.float32)

        for pos_idx, a_pos in enumerate(label_positions):
            if self.num_cls == 1 and len(a_pos) == 3:
                x, y, z = a_pos
                cls_idx_ = 1
            else:
                cls_idx_, x, y, z = a_pos

            if 'data_ocp' in self.label_type.lower():
                dim = int(self.label_diameter[cls_idx_ - 1])
            else:
                dim = int(self.label_diameter[0])
            radius = int(dim / 2)
            r = radius


            template = gaussian3D((dim, dim, dim), dim)

            cls_idx = pos_idx+1 if 'data_ocp' in self.label_type else cls_idx_
            # print(self.label_type, dim, cls_idx)
            z_start = 0 if z - r < 0 else z - r
            z_end = z_max if z + r + 1 > z_max else z + r + 1
            y_start = 0 if y - r < 0 else y - r
            y_end = y_max if y + r + 1 > y_max else y + r + 1
            x_start = 0 if x - r < 0 else x - r
            x_end = x_max if x + r + 1 > x_max else x + r + 1

            t_z_start = r - z if z - r < 0 else 0
            t_z_end = (r + z_max - z) if z + r + 1 > z_max else 2 * r + 1
            t_y_start = r - y if y - r < 0 else 0
            t_y_end = (r + y_max - y) if y + r + 1 > y_max else 2 * r + 1
            t_x_start = r - x if x - r < 0 else 0
            t_x_end = (r + x_max - x) if x + r + 1 > x_max else 2 * r + 1

            # print(z_start, z_end, y_start, y_end, x_start, x_end)
            # check border
            # print(label_data.shape)
            # print(z_start, z_end, y_start, y_end, x_start, x_end)
            tmp1 = label_data[z_start:z_end, y_start:y_end, x_start:x_end]
            tmp2 = template[t_z_start:t_z_end, t_y_start:t_y_end, t_x_start:t_x_end]

            larger_index = tmp1 < tmp2
            tmp1[larger_index] = tmp2[larger_index]

            tg = 0.60653  # 球形高斯核

            tmp1[tmp1 <= tg] = 0
            tmp1 = np.where(tmp1 > 0, cls_idx, 0)#使用粒子id来修改粒子高斯核
            label_data[z_start:z_end, y_start:y_end, x_start:x_end] = tmp1

        label_file.set_data(label_data)#把label_data写入到指定文件

        data_file.close()
        label_file.close()
        # print('work %s done' % i)
        # return 'work %s done' % i

    def gen_labels(self):
        if len(self.dir_list) == 1:
            self.single_handle(0)
        else:
            with Pool(len(self.dir_list)) as p:
                p.map(self.single_handle, np.arange(len(self.dir_list)).tolist())


def label_gen_show(args):
    base_path, coord_path, coord_format, tomo_path, tomo_format, \
    num_cls, label_type, label_diameter, stdout = args
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdou
        sys.stderr = stdout

    try:
        #生成路径和文件名
        label_gen = Coord_to_Label(base_path,
                                   coord_path,
                                   coord_format,
                                   tomo_path,
                                   tomo_format,
                                   num_cls,
                                   label_type,
                                   label_diameter)
        label_gen.gen_labels()
        if 'ocp' not in label_type:
            print('Label generation finished!')
            print('*' * 100)
        else:
            print('Occupancy generation finished!')
            print('*' * 100)
    except Exception as ex:
        term = 'Occupancy' if 'ocp' in label_type else 'Label'
        if stdout is not None:
            stdout.flush()
            stdout.write(f"{ex}")
            stdout.write(f'{term} Generation Exception!')
            print('*' * 100)
        else:
            traceback.print_exc()
            #print(f"{ex}")
            print(f'{term} Generation Exception!')
            print('*' * 100)
        return 0
    if stdout is not None:
        sys.stderr = save_stderr
        sys.stdout = save_stdout



if __name__ == "__main__":
    tmp1 = gaussian3D((5, 5, 5), 5)
    tmp2 = gaussian3D_weighted((5, 5, 5), 5)
    print(tmp1)
    print(tmp2)

    tmp1[tmp1 <= 0.223 ] = 0
    tmp1 = np.where(tmp1 > 0, 13, 0)
