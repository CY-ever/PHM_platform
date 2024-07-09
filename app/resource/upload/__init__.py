# 用户模块
from flask import Blueprint
from flask_restful import Api
from utils.contants import APP_URL_PREFIX
from utils.output_json import output_json

from app.resource.upload.file_upload import TimeDomainSignalUpload, LabelUpload, TrainDataUpload, TrainLabelUpload, \
    TestDataUpload, TestLabelUpload, DeleteUploadFile, SBN0DataUpload

# 创建蓝图对象
upload_bp = Blueprint("upload", __name__, url_prefix=APP_URL_PREFIX)
# 蓝图包装成api组件
upload_api = Api(upload_bp)
# 创建类视图(相应模块下)
# 给类视图添加路由信息<转换器名:接收变量名>

upload_api.add_resource(TimeDomainSignalUpload, "/upload/signal")
upload_api.add_resource(LabelUpload, "/upload/label")
upload_api.add_resource(TrainDataUpload, "/upload/traindata")
upload_api.add_resource(TrainLabelUpload, "/upload/trainlabel")
upload_api.add_resource(TestDataUpload, "/upload/testdata")
upload_api.add_resource(TestLabelUpload, "/upload/testlabel")
upload_api.add_resource(SBN0DataUpload, "/upload/sbn0")

upload_api.add_resource(DeleteUploadFile, "/upload/initialization")

# app中注册蓝图对象(init工厂方法中实现)
# 返回自定义的json字符串
upload_api.representation(mediatype="application/json")(output_json)
