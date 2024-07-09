import jwt
from flask import current_app


def generate_jwt(payload, expiry, secret=None):
    """
    生成jwt
    :param payload: dict 载荷
    :param expiry: datetime 有效期
    :param secret: 密钥
    :return: jwt
    """
    # 过期时间字典
    _payload = {'exp': expiry}
    # update字典更新添加键值对
    _payload.update(payload)
    """
    _payload = {
        "user_id": value,
        "exp": xxx
    }"""

    if not secret:
        # 从配置文件中读取配置信息
        secret = current_app.config['JWT_SECRET']

    token = jwt.encode(_payload, secret, algorithm='HS256')
    # TODO: 注意后期修改
    return token
    # return jwt.decode(token, secret, algorithms='HS256')


def verify_jwt(token, secret=None):
    """
    检验jwt
    :param token: jwt
    :param secret: 密钥
    :return: dict: payload
    """
    if not secret:
        secret = current_app.config['JWT_SECRET']

    try:
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        print(payload)
    except jwt.PyJWTError:
        payload = None

    # 返回的载荷字典可能为空
    return payload
