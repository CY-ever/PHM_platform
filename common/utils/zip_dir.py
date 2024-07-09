import os
import zipfile


def zip_dir(dirpath, out_fullname):
    """
    文件夹打包成压缩文件
    :param dirpath: 输入路径
    :param out_fullname: 输出路径
    :return:
    """
    zip = zipfile.ZipFile(out_fullname, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))

    zip.close()