from flask_sqlalchemy import SQLAlchemy


class DefaultConfig(object):
    """项目默认配置信息"""

    # session加密字符串
    SECRET_KEY = "PHM_TB"
    # 禁止flask_restful的json的ASCII编码
    RESTFUL_JSON = {"ensure_ascii" : False}

    # mysql连接配置(可注释掉,也可保留)
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:mysql@192.168.66.129:3306/PHMToolBox"
    # 多库联接信息
    SQLALCHEMY_BINDS = {
        # 主从数据库URI
        "master": "mysql+pymysql://root:mysql@192.168.66.129:3306/PHMToolBox",
        "slave1": "mysql+pymysql://root:mysql@192.168.66.129:3306/PHMToolBox",
        "slave2": "mysql+pymysql://root:mysql@192.168.66.129:3306/PHMToolBox",
    }

    # 关闭数据库修改跟踪
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True

    # redis连接配置
    # REDIS_HOST = "192.168.66.129"
    # REDIS_PORT = "6379"
    # redis哨兵联接信息
    SENTINEL_HOSTS = [
        {"192.168.66.129", 26380},
        {"192.168.66.129", 26381},
        {"192.168.66.129", 26382},
    ]
    SERVICE_NAME = "mymaster"    # 哨兵配置的主数据库别名

    # jwt密钥
    JWT_SECRET = "v432gtgtndm78tsydghn346165m87"
    # 6小时 和 14天 token
    JWT_LOGIN_EXPIRY = 24*30
    JWT_REFRESH_EXPIRY = 14

    # 允许跨域访问的前端 ip地址
    # CORS_ORIGINS = ["http://127.0.0.1:8000"]
    # CORS_ORIGINS = ["http://localhost:8000"]

class DevelopmentConfig(DefaultConfig):
    """开发环境配置信息"""
    DEBUG = True


class ProductionConfig(DefaultConfig):
    """生产环境配置信息"""
    DEBUG = False


# 外界调用接口
config_dict = {
    "dev": DevelopmentConfig,
    "pro": ProductionConfig
}