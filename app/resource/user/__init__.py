# 用户模块
from flask import Blueprint
from flask_restful import Api
from utils.contants import APP_URL_PREFIX
from utils.output_json import output_json
from app.resource.user.passport import SMSCodeResource, RegisterResource, LoginResource
from app.resource.user.profile import CurrentUserResource, UserUploadResource

# 创建蓝图对象
user_bp = Blueprint("user", __name__, url_prefix=APP_URL_PREFIX)
# 蓝图包装成api组件
user_api = Api(user_bp)
# 创建类视图(相应模块下)
# 给类视图添加路由信息<转换器名:接收变量名>
user_api.add_resource(SMSCodeResource, "/sms/code/<mob:mobil>")
user_api.add_resource(RegisterResource, "/user/register")
user_api.add_resource(LoginResource, "/user/authorization")
user_api.add_resource(CurrentUserResource, "/user")
user_api.add_resource(UserUploadResource, "/user/upload")
# app中注册蓝图对象(init工厂方法中实现)
# 返回自定义的json字符串
user_api.representation(mediatype="application/json")(output_json)
