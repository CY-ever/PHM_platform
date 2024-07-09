from flask import request, g, current_app
from utils.jwt_utils import verify_jwt


def get_userinfo():
    """每一次请求之前,调用该方法,统一提取token中的用户信息"""
    # 要求前端在请求头中携带token信息: {"Authorization": token}
    # 1. 提取请求头中的token信息
    token = request.headers.get("Authorization")
    secret = current_app.config["JWT_SECRET"]

    # 为用户信息设置默认值. 默认为未登录状态
    """
    # 无用户信息, 无token: 未登录状态
    g.user_id = None
    g.is_refresh = False
        
    # 已登录: 登录token---> 可直接进入视图函数
    g.user_id = 6
    g.is_refresh = True 
    
    # 已登录: 刷新token---> 需重新颁发一个登录token
    g.user_id = 6
    g.is_refresh = False
    """
    g.user_id = None
    g.is_refresh = False

    # 2. 校验token, 提取载荷信息
    if token:
        try:
            payload = verify_jwt(token, secret=secret)
        except Exception as ex:
            payload = None

        # 3. 从载荷字典中提取用户信息, 保存到g对象
        if payload:
            g.user_id = payload.get("user_id")
            g.is_refresh = payload.get("is_refresh", None)
