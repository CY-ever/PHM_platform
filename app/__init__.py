from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from redis import StrictRedis
from redis.sentinel import Sentinel
from app.setting.config import config_dict
import os, sys


# /common 路径添加到python: 这样再调用/common路径下的文件时,可直接调用,而不需要common.**
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH + "/common")
# 设置完路径资源后再导入
from utils.converters import register_converter
from utils.contants import EXTRA_ENV_CONFIG

from model.db_routing.routing_sqlalchemy import RoutingSQLAlchemy

# 创建数据库对象
# db = SQLAlchemy()
db = RoutingSQLAlchemy()
# redis_cli = None    # type: StrictRedis
# redis主客户端对象
redis_master = None
# redis从客户端对象
redis_slave = None

# 生产app
def create_flask_app(type):
    """内部调用app工厂方法: app初始化和环境配置"""

    # 创建app对象
    app = Flask(__name__)
    # 载入 配置类配置信息 和 环境变量配置信息
    class_name = config_dict[type]
    app.config.from_object(class_name)
    app.config.from_envvar(EXTRA_ENV_CONFIG, silent=True)

    return app


def create_app(type):
    """外部调用app工厂方法: 注册拓展组件 和 注册蓝图组件"""
    app = create_flask_app(type)

    # 注册拓展初始化组件
    register_extensions(app)
    # 注册蓝图初始化组件
    register_blueprint(app)
    return app


def register_extensions(app : Flask):
    """注册拓展初始化组件"""
    # 1. mysql数据库对象初始化
    db.init_app(app)
    # 2. redis数据库对象初始化
    # global redis_cli
    # redis_cli = StrictRedis(host=app.config["REDIS_HOST"],
    #                         port=app.config["REDIS_PORT"],
    #                         decode_responses=True,
    #                         db=0)
    # 2.1 创建哨兵客户端对象
    sentinel = Sentinel(app.config["SENTINEL_HOSTS"])
    global redis_master
    redis_master = sentinel.master_for(app.config["SERVICE_NAME"])
    global redis_slave
    redis_slave = sentinel.slave_for(app.config["SERVICE_NAME"])
    # 3. 给Flask添加自定义路由转换器
    register_converter(app)

    # 4. 数据库迁移
    Migrate(app, db)    # 一定还要导入执行迁移的模型文件!!
    from model import user
    """
    数据迁移指令:
    export FLASK_APP=app.main
    flask db init
    flask db migrate
    flask db upgrade
    # mysql指令,将指定.sql文件数据导入到制定数据库
    mysql -uroot -pmysql -h192.168.8.129 -D xxx数据库 < ./xxx.sql
    """
    # 5. 添加请求钩子装饰
    from utils.middleware import get_userinfo
    app.before_request(get_userinfo)

    # 6.支持跨与访问
    # oringins: default:* .默认允许所有前端地址都可访问
    # methode: default:[*], 允许的前端请求方式
    # supports_credentials: 支持跨与请求权限认证
    # CORS_ORIGINS = "http://192.168.1.1:8000"
    CORS(app, supports_credentials=True, )


def register_blueprint(app: Flask):
    # app中注册蓝图对象
    # 注意延迟导包!!!
    from app.resource.user import user_bp
    app.register_blueprint(user_bp)
    from app.resource.task import task_bp
    app.register_blueprint(task_bp)
    from app.resource.upload import upload_bp
    app.register_blueprint(upload_bp)


# if __name__ == '__main__':

    # redis 测试代码
    # app = create_app("dev")
    # redis_cli.set("key1", "123abc*")
    # print(redis_cli.get("key1"))