from clonevirtualenv import logger
from flask_restful import Resource
from flask_restful.reqparse import RequestParser
from flask_restful.inputs import *
from flask import current_app, g, abort

# from migrations.env import logger
from utils.contants import SMS_CODE_EXPIRE
from utils.jwt_utils import generate_jwt
from utils import parser as parser_tpye
from model.user import User, UserParameter
# from app import redis_cli
from app import db, redis_slave, redis_master
from sqlalchemy.orm import load_only
import random
import re


# 重置密码界面类视图
class SMSCodeResource(Resource):
    """发送短信验证码视图类"""

    def get(self, mobil):
        # 1. 产生一个随机6位验证码
        sms_code = "%06d" % (random.randint(0, 999999))
        sms_code = "123456"
        # 2. 将验证码保存在redis主库中
        # redis key 格式"app:code:"key""
        key = "app:code:{}".format(mobil)
        # redis_cli.setex(name=key, time=SMS_CODE_EXPIRE, value=sms_code)
        # redis读写分离
        try:
            redis_master.delete(key)
        except ConnectionError as e:
            logger.error(e)

        # 3. 使用第三方平台发送验证码(阿里云, 云通信, 腾讯云)
        print("短信验证码发送成功")
        # 4. 返回短信验证码,手机号码
        return {"mobil": mobil, "sms_code": sms_code}


# 注册
class RegisterResource(Resource):
    """
    注册类视图
    """

    def post(self):
        """
        1. 获取参数----获取用户注册各项个人信息
        2. 校验参数是否符合格式
        3. 逻辑处理----数据库增删改查
        4. 返回值处理----跳转回登录界面
        """
        # 1. 获取参数 - ---获取用户注册各项个人信息
        # 1.1 账号注册信息: 账号, 密码, 姓名, 性别, 生日, 简介
        # 2. 校验参数是否符合格式
        parser = RequestParser()
        parser.add_argument("account", required=True, location="json", type=parser_tpye.modul_parameter)
        parser.add_argument("password", required=True, location="json", type=parser_tpye.modul_parameter)
        # parser.add_argument("name", required=True, location="json", type=str)
        # parser.add_argument("gender", required=True, location="json", type=int)
        # parser.add_argument("birthday", required=True, location="json", type=parser_tpye.date)

        # TODO: user功能完善
        ret = parser.parse_args()
        account = ret["account"]
        password = ret["password"]
        # name = ret["name"]
        # gender = ret["gender"]
        # birthday = ret["birthday"]

        # 判断制定数据是否为空:如果为空,则报错
        # if not name:
        #     return {"message": "昵称不能为空哦:-)"}, 400

        # 3. 逻辑处理 - ---数据库增删改查
        print("len(account)", len(account))
        if len(account) > 40:
            abort(400, 'The maximum length of the registered mailbox is 40.')

        if re.match(r'^([A-Za-z0-9_\-\.\u4e00-\u9fa5])+\@([A-Za-z0-9_\-\.])+\.([A-Za-z]{2,8})$', account):
            pass
        else:
            abort(400, '{} is not a valid email'.format(account))

        if re.match(r'^[0-9A-Za-z\\W]{6,18}$', password):
            pass
        else:
            abort(400, '{} is not a valid password'.format(password))

        # 3.1 判断账户是否已经存在
        db_account = User.query.options(load_only(User.user_account)).filter(User.user_account == account).first()
        if db_account:
            return {"message": "The account already exists."}, 400

        # TODO: user功能完善
        # 3.2 如该账号尚未被注册,则将信息写入数据库
        new_user = User(user_account=account,
                        user_password = password,
                        # user_name=name,
                        # user_gender=gender,
                        # user_birthday=birthday,
                        last_login=datetime.now())

        db.session.add(new_user)
        try:
            db.session.commit()
        except Exception as ex:
            db.session.rollback()
            return {"message" : ex}, 507

        # 4. 返回值处理 - ---跳转回登录界面
        return "注册成功"


# 登录
class LoginResource(Resource):
    """
    登录类视图
    """
    def __generator_token(self, user_id):
        """
        生成一个2小时登录token, 和14天刷新token(在登录token过期后,重新生成一个2小时登录token)
        :return: login_token, refresh_token
        """
        # 1. 生成2小时有效登录token
        # 准备载荷
        login_payload = {
            "user_id" : user_id,
            "is_refresh" : False
        }
        # 准备2小时过期时常
        expiry_2h = datetime.utcnow() + timedelta(hours=current_app.config["JWT_LOGIN_EXPIRY"])
        # 从配置信息中获取密钥
        secret = current_app.config["JWT_SECRET"]
        # 生成2h_token
        login_token = generate_jwt(payload=login_payload, expiry=expiry_2h, secret=secret)

        # 2. 生成14天 刷新token: 同上
        # 准备载荷
        refresh_payload = {
            "user_id": user_id,
            "is_refresh": True
        }
        # 准备2小时过期时常
        expiry_14days = datetime.utcnow() + timedelta(days=current_app.config["JWT_REFRESH_EXPIRY"])
        # 生成2h_token
        refresh_token = generate_jwt(payload=refresh_payload, expiry=expiry_14days, secret=secret)

        # # 3. 修改用户最后登录时间为当前时间
        # user = User.query.options(load_only(User.last_login)).filter(User.user_id == user_id).first()
        # user.last_login = datetime.now()

        return login_token, refresh_token
        # return login_token

    def post(self):
        """
        1. 获取参数----获取 账号(邮箱) 密码
        2. 校验参数是否符合格式
        3. 逻辑处理----数据库提取用户账号密码进行比对验证
        4. 返回值处理----返回6小时登录token和14天刷新token
        """
        # 1. 获取参数----获取 账号 密码
        # 2. 校验参数是否符合格式

        print("123456")
        parser = RequestParser()
        parser.add_argument("account", required=True, location="json", type=parser_tpye.email)
        parser.add_argument("password", required=True, location="json", type=parser_tpye.password)
        print("123456")

        ret = parser.parse_args()
        account = ret["account"]
        password = ret["password"]

        print("账号", account)
        print("密码", password)



        # 3. 逻辑处理----数据库提取用户账号密码进行比对验证
        # 3.1 获取真实账户密码----从mysql数据库中提取
        # 3.2 判断账号是否存在: 如账号不存在 抛出异常
        # 3.3 判断密码是否正确: 如错误 抛出异常
        user = User.query.options(load_only(User.user_account, User.user_password)).filter(User.user_account == account).first()

        if user is None or user.user_password != password:
            return {"message": "密码错误或账号不存在"}, 400

        # 返回值处理----返回6小时登录token和14天刷新token
        login_token, refresh_token = self.__generator_token(user.user_id)
        # return {"login_token": login_token, "refresh_token": refresh_token}
        return {"message": "OK",
                "login_token": login_token,
                "refresh_token": refresh_token
                }

    def put(self):
        """刷新token有效期内,给予新登录token"""
        user_id = g.user_id
        # 当token为刷新token,而不是登录token时,生成新登录token返回
        if user_id and g.is_refresh is True:
            login_token, useless_token = self.__generator_token(user_id)
            return {"new_token": login_token}


