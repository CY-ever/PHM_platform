from __future__ import absolute_import
import datetime
from flask import make_response, current_app, request
from flask_restful.utils import PY3
import json
from json import dumps


# 重写dumps的JSONEncoder, 使datatime型自动转换成字符串 ==> output_json使用
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            print("JSON:", self,obj)
            return json.JSONEncoder.default(self,obj)


def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""
    if str(code) == 400:
        current_app.logger.warn(request.headers)
        current_app.logger.warn(request.data)
        current_app.logger.warn(str(data))

    if 'message' not in data:
        data = {
            'message': 'OK',
            'data': data
        }

    settings = current_app.config.get('RESTFUL_JSON', {})

    # If we're in debug mode, and the indent is not set, we set it to a
    # reasonable value here.  Note that this won't override any existing value
    # that was set.  We also set the "sort_keys" value.
    if current_app.debug:
        settings.setdefault('indent', 4)
        settings.setdefault('sort_keys', not PY3)

    # always end the json dumps with a new line
    # see https://github.com/mitsuhiko/flask/pull/1262
    dumped = dumps(data, **settings, cls=DateEncoder) + "\n"

    resp = make_response(dumped, code)
    resp.headers.extend(headers or {})
    return resp
