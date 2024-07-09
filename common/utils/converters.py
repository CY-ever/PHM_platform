from werkzeug.routing import BaseConverter


class MobilConverter(BaseConverter):
    """
    手机号码格式
    """
    regex = r'1[3-9]\d{9}'


def register_converter(app):
    """在Flask app中添加转换器"""

    app.url_map.converters['mob'] = MobilConverter
