# 项目启动文件
from app import create_app
from flask import jsonify, Flask

app = create_app("dev")
# app = Flask(__name__)
# 创建app对象




@app.route("/")
def index():
    # 返回所有陆游信息
    route_data = {rule.rule:rule.endpoint for rule in app.url_map.iter_rules()}
    return jsonify(route_data)


@app.route("/user")
def user():

    return "个人界面"
