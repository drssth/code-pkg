import os
import os.path
import glob


def get_files(path, pattern='*.npz'):
    result = [y for x in os.walk(path)
              for y in glob.glob(os.path.join(x[0], pattern))]
    return result


def get_subdir_list(data_path, item=None):
    ls = os.listdir(data_path)
    r = {}
    for i in ls:
        if item is None:
            if '.' not in i:
                p_path = os.path.join(data_path, i)
                if os.path.isdir(p_path):
                    r[i] = p_path
        else:
            p_path = os.path.join(data_path, i, item)
            if os.path.exists(p_path):
                r[i] = p_path
    return r
