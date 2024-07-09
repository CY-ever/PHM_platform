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


class CurrentUserResource(Resource):
    """获取当前登录用户信息 类视图"""

    # 添加装饰器
    method_decorators = {
        "get": [login_required]
    }

    def get(self):

        # TODO: 访问数据库, 提取页面相关.并包装成字典返回
        # 例:
        user_id = g.user_id
        # 从数据库提取用户数据
        user = User.query.options(load_only(User.user_id,
                                            User.user_account,
                                            User.user_gender,
                                            User.user_name,
                                            User.user_introduction)).filter(User.user_id == user_id).first()
        print(user)
        # 转成字典
        user_dict = {}
        if user:
            user_dict = user.to_dict()

        print(user_dict)
        user_dict.pop("user_password")
        # return {"data": "欢迎回来!"}
        return  user_dict


class UserUploadResource(Resource):
    """用户上传数据类视图"""
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }
    """
        # 1.获取参数
        # 1.1 数据文件
        # 1.2 user_id通过g对象获取
        # 2.校验参数
        # 3.逻辑处理
        # 3.1 文件对象转换成二进制
        # 3.2 上传到七牛云
        # 3.3 根据user_id将图片的url地址保存到该用户的file_url属性下
        # 或:
        # 3.1 文件名重命名,并保存在本地,
        # 4.返回值处理 
    """
    def post(self):
        """
        修改用户的资料（修改用户头像）
        :return:
        """
        # 1.获取请求参数
        # 2.校验请求参数
        # TODO: 校验文件格式:.zip/.7z/exl
        rp = RequestParser()
        # 1.1 数据文件
        rp.add_argument("data", type=parser.image_file, required=True, location='files')
        ret = rp.parse_args()
        data_file = ret["data"]     # type:FileStorage
        # 1.2 user_id通过g对象获取
        user_id = g.user_id

        # 3.逻辑处理
        # # 3.1 文件对象转换成二进制
        # file_binary = data_file.read()
        # # 3.2 上传到七牛云
        # # req.photo 取出了请求中的文件对象，通过read方法读取文件的二进制数据
        # try:
        #     full_url = upload(file_binary)
        # except Exception as e:
        #     return {"message":"文件上传七牛云失败:{}".format(e)}
        # # 3.3 根据user_id将图片的url地址保存到该用户的file_url属性下
        # # url = 域名 + / + 文件名称
        # # 所以,可以光保存文件名称
        # # TODO:尚未添加User表格url字段
        # User.query.filter(User.id == user_id).updata({"":full_url})
        # try:
        #     db.session.commit()
        # except Exception as e:
        #     db.session.rollback()
        #     return {"message":"提交文件失败:{}".format(e)}
        ################################
        # 或:
        # 3.1 文件转成二进制,保存在本地对应用户id文件夹下的.txt文件里
        # file_binary = data_file.read()
        dir_name = f"./user_file/{user_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = f"./user_file/{user_id}/upload"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        data_file.save(os.path.join(dir_name, data_file.filename))
        data_file.close()
        # if '.mat' in data_file.filename:
        #     data_file.save(os.path.join(dir_name, data_file.filename))
            # fb = open(os.path.join(dir_name, data_file.filename), mode='bw')
            # fb.write(file_binary)
        # else:
        #     fb = open(os.path.join(dir_name, data_file.filename), mode='w', encoding='utf-8')
        #     fb.write(file_binary.decode())
        #     fb.write(file_binary)
        #     fb.close()

        return {"message":"file upload successfully!"}






