from flask import g, current_app, request
from flask_restful import Resource
from flask_restful.reqparse import RequestParser
from utils import parser
from utils.qiniu_storage import upload
from utils.decorators import login_required
from sqlalchemy.orm import load_only

from app import db
from model.user import User
from werkzeug.datastructures import FileStorage
import os
import shutil

"""
    各个模块的上传文件功能分别独立:
    模块一: 上传时域信号(nums, x)
    模块二: 上传时域信号(nums, x)
    模块三: 降噪部分:上传时域信号(nums, x)//数据切分部分:上传时域信号 与 label (双文件)(nums, x)(nums, 1)
    模块四: 特征提取/特征选择:上传时域信号 与 label (双文件)(nums, x)(nums, 1)
    模块五: 训练集,测试集,训练label,测试label (四文件)(nums, x)(nums, 1)

    综上所述:
    接口一: time domain signal(no label) 单文件上传
    接口二: label 文件上传
    接口三: train_data
    接口四: train_label
    接口五: test_data
    接口六: test_label
"""

# class ReferenceSampleUpload(Resource):
#     """
#     为GMR_FFT上传参考样本
#     """
#     # 添加装饰器
#     method_decorators = {
#         "post": [login_required]
#     }
#
#     def post(self):
#         """
#         修改用户的资料（修改用户头像）
#         :return:
#         """
#         print("开始上传文件")
#         # 1.获取请求参数
#         # 2.校验请求参数
#         # TODO: 校验文件格式:.zip/.7z/exl
#         rp = RequestParser()
#         # 1.1 获取数据文件
#         rp.add_argument("data", type=parser.image_file, required=True, location='files')
#         ret = rp.parse_args()
#         data_file = ret["data"]  # type:FileStorage
#         # 1.2 user_id通过g对象获取
#         user_id = g.user_id
#
#         # 2. 逻辑处理
#         # 2.1 文件类型判断
#         if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
#                 ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
#                 ".csv" not in data_file.filename):
#             return {"message": "The file type is not supported"}, 405
#
#         # 2.2 老文件删除
#         try:
#             file_list = os.listdir(f"./upload_file/{user_id}/upload")
#         except:
#             os.makedirs(f"./upload_file/{user_id}/upload")
#             file_list = os.listdir(f"./upload_file/{user_id}/upload")
#         print("file_list", file_list)
#         for name in file_list:
#             if "time_domain_signal" in name:
#                 os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
#                 break
#
#         # 2.3 文件命名
#         dir_name = f"./upload_file/{user_id}"
#         if not os.path.exists(dir_name):
#             os.mkdir(dir_name)
#         dir_name = f"./upload_file/{user_id}/upload"
#         if not os.path.exists(dir_name):
#             os.mkdir(dir_name)
#
#         file_type = data_file.filename.split('.')[-1]
#         print("file_type", file_type)
#
#         data_file.filename = f"time_domain_signal.{file_type}"
#         # 3.文件保存
#         data_file.save(os.path.join(dir_name, data_file.filename))
#         data_file.close()
#
#         return {"message": "file upload successfully!"}

class TimeDomainSignalUpload(Resource):
    """
    用户上传数据类视图: Modul_1 and Modul_2 and Moudul3
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        print("开始上传文件")
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]  # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # 2. 逻辑处理
        # 2.1 文件类型判断
        if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
                ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
                ".csv" not in data_file.filename):
            return {"message": "The file type is not supported"}, 405

        # 2.2 老文件删除
        try:
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        except:
            os.makedirs(f"./upload_file/{user_id}/upload")
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        print("file_list", file_list)
        for name in file_list:
            if "time_domain_signal" in name:
                os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
                break

        # 2.3 文件命名
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./upload_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_type = data_file.filename.split('.')[-1]
        print("file_type", file_type)

        data_file.filename = f"time_domain_signal.{file_type}"
        # 3.文件保存
        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()

        return {"message": "File upload successfully!"}


class LabelUpload(Resource):
    """
    用户上传数据类视图: Modul_1 and Modul_2 and Moudul3
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]  # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # # 2. 逻辑处理
        # # 2.1 文件类型判断
        # if ".mat" not in data_file.filename:
        #     return {"message": "The file type is not supported!"}, 405
        # # 2.2 文件命名
        # dir_name = f"./upload_file/{user_id}"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        # dir_name = f"./upload_file/{user_id}/upload"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        #
        # data_file.filename = "data_label.mat"
        # # 3.文件保存
        # data_file.save(os.path.join(dir_name, data_file.filename))
        # data_file.close()
        #
        # return {"message": "File upload successfully!"}
        # 2. 逻辑处理
        # 2.1 文件类型判断
        if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
                ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
                ".csv" not in data_file.filename):
            return {"message": "The file type is not supported"}, 405

        # 2.2 老文件删除
        try:
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        except:
            os.makedirs(f"./upload_file/{user_id}/upload")
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        print("file_list", file_list)
        for name in file_list:
            if "data_label" in name:
                os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
                break

        # 2.3 文件命名
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./upload_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_type = data_file.filename.split('.')[-1]
        print("file_type", file_type)

        data_file.filename = f"data_label.{file_type}"
        # 3.文件保存
        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()

        return {"message": "File upload successfully!"}


class TrainDataUpload(Resource):
    """
    用户上传数据类视图: Modul_1 and Modul_2 and Moudul3
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]  # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # # 2. 逻辑处理
        # # 2.1 文件类型判断
        # if ".mat" not in data_file.filename:
        #     return {"message": "The file type is not supported!"}, 405
        # # 2.2 文件命名
        # dir_name = f"./upload_file/{user_id}"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        # dir_name = f"./upload_file/{user_id}/upload"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        #
        # data_file.filename = "traindata.mat"
        # # 3.文件保存
        # data_file.save(os.path.join(dir_name, data_file.filename))
        # data_file.close()
        #
        # return {"message": "File upload successfully!"}
        # 2. 逻辑处理
        # 2.1 文件类型判断
        if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
                ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
                ".csv" not in data_file.filename):
            return {"message": "The file type is not supported"}, 405

        # 2.2 老文件删除
        try:
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        except:
            os.makedirs(f"./upload_file/{user_id}/upload")
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        print("file_list", file_list)
        for name in file_list:
            if "traindata" in name:
                os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
                break

        # 2.3 文件命名
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./upload_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_type = data_file.filename.split('.')[-1]
        print("file_type", file_type)

        data_file.filename = f"traindata.{file_type}"
        # 3.文件保存
        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()

        return {"message": "File upload successfully!"}


class TrainLabelUpload(Resource):
    """
    用户上传数据类视图: Modul_1 and Modul_2 and Moudul3
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]  # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # # 2. 逻辑处理
        # # 2.1 文件类型判断
        # if ".mat" not in data_file.filename:
        #     return {"message": "The file type is not supported!"}, 405
        # # 2.2 文件命名
        # dir_name = f"./upload_file/{user_id}"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        # dir_name = f"./upload_file/{user_id}/upload"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        #
        # data_file.filename = "trainlabel.mat"
        # # 3.文件保存
        # data_file.save(os.path.join(dir_name, data_file.filename))
        # data_file.close()
        #
        # return {"message": "File upload successfully!"}
        # 2. 逻辑处理
        # 2.1 文件类型判断
        if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
                ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
                ".csv" not in data_file.filename):
            return {"message": "The file type is not supported"}, 405

        # 2.2 老文件删除
        try:
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        except:
            os.makedirs(f"./upload_file/{user_id}/upload")
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        print("file_list", file_list)
        for name in file_list:
            if "trainlabel" in name:
                os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
                break

        # 2.3 文件命名
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./upload_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_type = data_file.filename.split('.')[-1]
        print("file_type", file_type)

        data_file.filename = f"trainlabel.{file_type}"
        # 3.文件保存
        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()

        return {"message": "File upload successfully!"}


class TestDataUpload(Resource):
    """
    用户上传数据类视图: Modul_1 and Modul_2 and Moudul3
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]  # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # # 2. 逻辑处理
        # # 2.1 文件类型判断
        # if ".mat" not in data_file.filename:
        #     return {"message": "The file type is not supported!"}, 405
        # # 2.2 文件命名
        # dir_name = f"./upload_file/{user_id}"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        # dir_name = f"./upload_file/{user_id}/upload"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        #
        # data_file.filename = "testdata.mat"
        # # 3.文件保存
        # data_file.save(os.path.join(dir_name, data_file.filename))
        # data_file.close()
        #
        # return {"message": "File upload successfully!"}
        # 2. 逻辑处理
        # 2.1 文件类型判断
        if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
                ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
                ".csv" not in data_file.filename):
            return {"message": "The file type is not supported"}, 405

        # 2.2 老文件删除
        try:
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        except:
            os.makedirs(f"./upload_file/{user_id}/upload")
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        print("file_list", file_list)
        for name in file_list:
            if "testdata" in name:
                os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
                break

        # 2.3 文件命名
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./upload_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_type = data_file.filename.split('.')[-1]
        print("file_type", file_type)

        data_file.filename = f"testdata.{file_type}"
        # 3.文件保存
        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()

        return {"message": "File upload successfully!"}


class TestLabelUpload(Resource):
    """
    用户上传数据类视图: Modul_1 and Modul_2 and Moudul3
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]  # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # # 2. 逻辑处理
        # # 2.1 文件类型判断
        # if ".mat" not in data_file.filename:
        #     return {"message": "The file type is not supported!"}, 405
        # # 2.2 文件命名
        # dir_name = f"./upload_file/{user_id}"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        # dir_name = f"./upload_file/{user_id}/upload"
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        #
        # data_file.filename = "testlabel.mat"
        # # 3.文件保存
        # data_file.save(os.path.join(dir_name, data_file.filename))
        # data_file.close()
        #
        # return {"message": "File upload successfully!"}
        # 2. 逻辑处理
        # 2.1 文件类型判断
        if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
                ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
                ".csv" not in data_file.filename):
            return {"message": "The file type is not supported"}, 405

        # 2.2 老文件删除
        try:
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        except:
            os.makedirs(f"./upload_file/{user_id}/upload")
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        print("file_list", file_list)
        for name in file_list:
            if "testlabel" in name:
                os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
                break

        # 2.3 文件命名
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./upload_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_type = data_file.filename.split('.')[-1]
        print("file_type", file_type)

        data_file.filename = f"testlabel.{file_type}"
        # 3.文件保存
        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()

        return {"message": "File upload successfully!"}


class DeleteUploadFile(Resource):
    """
        重置上传文件
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件

        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # 2. 逻辑处理
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        try:
            shutil.rmtree(f"/home/python/Desktop/PHM_Dev/upload_file/{user_id}/upload")
            os.makedirs(f"/home/python/Desktop/PHM_Dev/upload_file/{user_id}/upload")

        except:
            os.makedirs(f"/home/python/Desktop/PHM_Dev/upload_file/{user_id}/upload")

        return {"message": "Upload file successfully initialized!"}


class SBN0DataUpload(Resource):
    """
    上传SBN0数据
    """
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        print("开始上传文件")
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 获取数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]  # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # 2. 逻辑处理
        # 2.1 文件类型判断
        if (".mat" not in data_file.filename) and (".xls" not in data_file.filename) and (
                ".npy" not in data_file.filename) and (".txt" not in data_file.filename) and (
                ".csv" not in data_file.filename):
            return {"message": "The file type is not supported"}, 405

        # 2.2 老文件删除
        try:
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        except:
            os.makedirs(f"./upload_file/{user_id}/upload")
            file_list = os.listdir(f"./upload_file/{user_id}/upload")
        print("file_list", file_list)
        for name in file_list:
            if "time_domain_signal" in name:
                os.remove(os.path.join(f"./upload_file/{user_id}/upload", name))
                break

        # 2.3 文件命名
        dir_name = f"./upload_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./upload_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        file_type = data_file.filename.split('.')[-1]
        print("file_type", file_type)

        data_file.filename = f"SBN0.{file_type}"
        # 3.文件保存
        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()

        return {"message": "File upload successfully!"}