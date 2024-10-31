from datetime import datetime
from pathlib import Path
import numpy as np
from utils.file.file_util import get_root_dir

rootdir = Path(get_root_dir())
# 保存二维数组到文件
def save_array(array, filename):
    filename = rootdir / 'data' / 'result' / filename
    np.savetxt(filename, array)

# 从文件中读取二维数组
def load_array(filename):
    filename = rootdir / 'data' / 'result' / filename
    array = np.loadtxt(filename)
    return array

if __name__ == '__main__':

    # # 示例二维数组
    # array = np.array([[1, 1, 3], [4, 5, 6], [7, 8, 9]])
    # file  = datetime.today().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    # # 保存二维数组到文件，覆盖已有文件
    # save_array(array, file)

    # 从文件中读取二维数组
    loaded_array = load_array('array.txt')
    print("Loaded array:")
    print(loaded_array)