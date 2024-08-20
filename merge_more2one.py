import os
import re
import torch

if __name__ == '__main__':

    import sys
    folder = sys.argv[1]
    dst_file = sys.argv[2]

    assert os.path.exists(folder), f"{folder} not exists."
    assert dst_file.endswith('pt'), "expected postfix is pt."

    files = [os.path.join(folder, y) for y in sorted(os.listdir(folder), key = lambda x: int(''.join(re.findall('[0-9]+', x))))]
    attrs_list = []

    for _file in files:
        attr_list = torch.load(_file)
        if isinstance(attr_list, list):
            x,y = attr_list
            x = x.cpu()
            y = [_y.cpu() for _y in y]
            attrs_list.append([x, y])
        else:
            attrs_list.append(attr_list)
    assert len(attrs_list) >0, "no file has been processed."
    torch.save(attrs_list, dst_file)
