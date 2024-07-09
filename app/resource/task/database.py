from flask import g, current_app, request, Response
from flask_restful import Resource
from flask_restful.reqparse import RequestParser
from utils.decorators import login_required
from utils.database_opt import inputlength
from utils import parser
from flask import abort
import glob


class CWRU(Resource):
    """
    获取结果图片
    """

    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        # 参数校验,参数获取
        # text = request.json
        # print(request.json)
        rp = RequestParser()
        rp.add_argument("file_lists", type=parser.modul_parameter, required=True, location="json")

        ret = rp.parse_args()
        file_lists = ret["file_lists"]

        # 例:
        user_id = g.user_id

        # 参数处理
        from moduls.utils.utils import str_to_int
        file_lists = str_to_int(file_lists)
        print("参数为:", file_lists)

        #
        label0 = (157, 158, 159, 160)
        label3 = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 112,
            113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123)
        label2 = (
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
            83, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135)
        label1 = (
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            51,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
            154,
            155, 156)

        switch = [0, 0, 0, 0]
        for i in file_lists:
            if i in label0:
                pass
            elif i in label1:
                switch[0] = 1
            elif i in label2:
                switch[1] = 1
            elif i in label3:
                switch[2] = 1
            else:
                abort(400, f"Bearing number {i} does not exist.")
        # CWRU
        inputl = inputlength(fs=12000,
                             fr=1797 / 60,
                             order=8,
                             switch=switch,
                             n_ball=9,
                             d_ball=7.94,
                             d_pitch=39.04)

        return {"message": "The optimal sample length is %d." % (inputl)}, 400


class Parderborn(Resource):
    """
    获取结果图片
    """

    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        # 参数校验,参数获取
        # text = request.json
        # print(request.json)
        rp = RequestParser()
        rp.add_argument("file_lists", type=parser.modul_parameter, required=True, location="json")

        ret = rp.parse_args()
        file_lists = ret["file_lists"]

        print("参数为:", file_lists)

        # 例:
        user_id = g.user_id

        # 参数处理
        from moduls.utils.utils import str_to_int
        file_lists = str_to_int(file_lists)
        print("参数为:", file_lists)

        #
        label0 = (27, 28, 29, 30, 31, 32, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126, 127, 128)
        label1 = (
            1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37, 47, 48, 49, 50, 51, 52, 53, 65, 66, 67, 68,
            69, 79, 80, 81, 82, 83, 84, 85, 97, 98, 99, 100, 101, 111, 112, 113, 114, 115, 116, 117)
        label2 = (
            9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26, 41, 42, 43, 44, 45, 46, 54, 55, 56, 57, 58, 73, 74, 75, 76, 77,
            78, 86, 87, 88, 89, 90, 105, 106, 107, 108, 109, 110, 118, 119, 120, 121, 122)

        label12 = (6, 7, 8, 38, 39, 40, 70, 71, 72, 102, 103, 104)

        switch = [0, 0, 0, 0]
        for i in file_lists:
            if i in label0:
                pass
            elif i in label1:
                switch[0] = 1
            elif i in label2:
                switch[1] = 1
            elif i in label12:
                switch[0] = 1
                switch[1] = 1
            else:
                abort(400, f"Bearing number {i} does not exist.")

        # Parderborn
        inputl = inputlength(fs=16000,
                             fr=1500 / 60,
                             order=8,
                             switch=switch,
                             n_ball=8,
                             d_ball=6.75,
                             d_pitch=29.05)


        return {"message": "The optimal sample length is %d." % (inputl)}, 400


class XJTU(Resource):
    """
    获取结果图片
    """

    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        rp = RequestParser()
        rp.add_argument("file_lists", type=parser.modul_parameter, required=True, location="json")

        ret = rp.parse_args()
        file_lists = ret["file_lists"]

        # 例:
        user_id = g.user_id

        # 参数处理
        from moduls.utils.utils import str_to_int
        file_lists = str_to_int(file_lists)
        print("参数为:", file_lists)

        #
        label1 = (1, 2, 3, 4, 5, 6, 13, 14, 17, 18, 19, 20, 21, 22, 29, 30)
        label2 = (11, 12, 25, 26, 27, 28)
        label4 = (7, 8, 15, 16)
        label1234 = (23, 24)
        label12 = (9, 10)

        switch = [0, 0, 0, 0]
        for i in file_lists:
            if i in label1:
                switch[0] = 1
            elif i in label2:
                switch[1] = 1
            elif i in label4:
                switch[3] = 1
            elif i in label12:
                switch[0] = 1
                switch[1] = 1
            elif i in label1234:
                switch[0] = 1
                switch[1] = 1
                switch[2] = 1
                switch[3] = 1
            else:
                abort(400, f"Bearing number {i} does not exist.")

        # fr判断
        fr_35 = [1,2,3,4,5,6,7,8, 9, 10]
        fr_375 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        fr_42 = [21, 22, 23, 24, 25, 26,27,28,29,30]
        fr = 35
        for i in file_lists:
            if i in fr_35:
                fr = 35
            elif i in fr_375:
                fr = 37.5
            elif i in fr_42:
                fr = 42
            else:
                fr = 35

        # XJTU
        inputl = inputlength(fs=25600,
                             fr=fr,
                             order=8,
                             switch=switch,
                             n_ball=8,
                             d_ball=7.92,
                             d_pitch=34.55)

        return {"message": "The optimal sample length is %d." % (inputl)}, 200


class IMS(Resource):
    """
    获取结果图片
    """

    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        # 参数校验,参数获取
        # text = request.json
        # print(request.json)
        rp = RequestParser()
        rp.add_argument("file_lists", type=parser.modul_parameter, required=True, location="json")

        ret = rp.parse_args()
        file_lists = ret["file_lists"]

        print("参数为:", file_lists)

        # 例:
        user_id = g.user_id

        # 参数处理
        from moduls.utils.utils import str_to_int
        file_lists = str_to_int(file_lists)
        print("参数为:", file_lists)

        #
        label0 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        label1 = (11, 12)
        label2 = (13, 14)
        label3 = (15, 16)

        switch = [0, 0, 0, 0]

        for i in file_lists:
            if i in label0:
                pass
            elif i in label1:
                switch[0] = 1
            elif i in label2:
                switch[1] = 1
            elif i in label3:
                switch[2] = 1
            else:
                abort(400, f"Bearing number {i} does not exist.")

        # IMS
        inputl = inputlength(fs=20000,
                             fr=2000 / 60,
                             order=8,
                             switch=switch,
                             n_ball=16,
                             d_ball=8.4,
                             d_pitch=71.5)

        return {"message": "The optimal sample length is %d." % (inputl)}, 200


class FEMTO(Resource):
    """
    获取结果图片
    """

    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }

    def post(self):
        # 参数校验,参数获取
        # text = request.json
        # print(request.json)
        rp = RequestParser()
        rp.add_argument("file_lists", type=parser.modul_parameter, required=True, location="json")

        ret = rp.parse_args()
        file_lists = ret["file_lists"]

        print("参数为:", file_lists)

        # 例:
        user_id = g.user_id

        # 参数处理
        from moduls.utils.utils import str_to_int
        file_lists = str_to_int(file_lists)
        print("参数为:", file_lists)

        return {"message": "The database does not require optimization."}, 200
