# 用户模块
from flask import Blueprint
from flask_restful import Api
from utils.contants import APP_URL_PREFIX
from utils.output_json import output_json
from app.resource.task.upload import Download, StartTasks
# import app.resource.task.upload
from app.resource.task.profile import ResultsImage
from app.resource.task.database import CWRU, XJTU, FEMTO, Parderborn, IMS

# 创建蓝图对象
task_bp = Blueprint("task", __name__, url_prefix=APP_URL_PREFIX)
# 蓝图包装成api组件
task_api = Api(task_bp)
# 创建类视图(相应模块下)
# 给类视图添加路由信息<转换器名:接收变量名>
task_api.add_resource(StartTasks, "/tasks/submit")
task_api.add_resource(Download, "/tasks/download")
task_api.add_resource(ResultsImage, "/tasks/results")

task_api.add_resource(CWRU, "/tasks/cwru")
task_api.add_resource(XJTU, "/tasks/xjtu")
task_api.add_resource(Parderborn, "/tasks/paderborn")
task_api.add_resource(IMS, "/tasks/ims")
task_api.add_resource(FEMTO, "/tasks/femto")

# app中注册蓝图对象(init工厂方法中实现)
# 返回自定义的json字符串
task_api.representation(mediatype="application/json")(output_json)