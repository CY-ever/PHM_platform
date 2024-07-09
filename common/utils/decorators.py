import functools

from flask import g

"""
业务逻辑:
1. user_id = None ==> 未登录
2. user_id = 6, refresh = True ==> 刷新token ==> 访问刷新接口
3. user_id = 6, refresh = False ==> 登录token ==> 允许进入视图函数
"""


# 强制登录装饰器
def login_required(view_func):
    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        user_id = g.user_id
        # 根据 业务逻辑, 判断当前用户的登录状态
        # 401 权限认证失败 unauthorization
        if user_id is None:
            return {"message": "invalid token"}, 401
        # 403 刷新token 不能用作用户权限认证
        elif user_id and g.is_refresh is True:
            return {"message": "do not use refresh token for authorization"}, 403
        else:
            # 登录token, 允许访问视图函数
            return view_func(*args, **kwargs)

    return wrapper
