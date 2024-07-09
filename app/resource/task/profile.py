from flask import g, current_app, request, Response
from flask_restful import Resource
from flask_restful.reqparse import RequestParser
from utils.decorators import login_required
from utils import parser
from flask import abort
import glob


class ResultsImage(Resource):
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
        print(request.json)
        rp = RequestParser()
        rp.add_argument("task_code", type=parser.modul_parameter, required=True, location="json")

        ret = rp.parse_args()
        task_code = ret["task_code"]

        print("标识参数为:", task_code)

        # 例:
        user_id = g.user_id

        try:
            # 逻辑判断:根据不同的标识符,返回指定图片
            if task_code == 11:
                # file_list = glob.glob(rf'user_file/{user_id}/signal_based_bearing_defect_model.*')
                # print(file_list)
                # with open(file_list[0], 'rb') as file:
                try:
                    with open(f"user_file/{user_id}/time and frequency domain acceleration0.png", 'rb') as file:
                        image = file.read()
                except:
                    try:
                        with open(f"user_file/{user_id}/time and frequency domain acceleration1.png", 'rb') as file:
                            image = file.read()
                    except:
                        with open(f"user_file/{user_id}/time and frequency domain acceleration2.png", 'rb') as file:
                            image = file.read()

            elif task_code == 12:
                # file_list = glob.glob(fr'user_file/{user_id}/signal_based_bearing_defect_model.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/signal_based_bearing_defect_model1.png", 'rb') as file:
                    image = file.read()
            elif task_code == 21:
                # file_list = glob.glob(fr'user_file/{user_id}/transformation_DA_image_new1.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/transformation_DA_image_new2.png", 'rb') as file:
                    image = file.read()
            elif task_code == 22:
                # file_list = glob.glob(fr'user_file/{user_id}/GAN_image1.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/GAN_image2.png", 'rb') as file:
                    image = file.read()
            elif task_code == 23:
                # file_list = glob.glob(fr'user_file/{user_id}/GAN_image1.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/Degradation_trajectories_after_Monte_Carlo_sampling.png", 'rb') as file:
                    image = file.read()
            elif task_code == 31:
                # file_list = glob.glob(fr'user_file/{user_id}/FFT.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/FFT.png", 'rb') as file:
                    image = file.read()
            elif task_code == 32:
                # file_list = glob.glob(fr'user_file/{user_id}/DWT.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/DWT.png", 'rb') as file:
                    image = file.read()
            elif task_code == 33:
                # file_list = glob.glob(fr'user_file/{user_id}/EMD_filtered_data.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/EMD_filtered_data.png", 'rb') as file:
                    image = file.read()
            elif task_code == 34:
                # file_list = glob.glob(fr'user_file/{user_id}/mean_filter_data.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file//{user_id}/mean_filter_data.png", 'rb') as file:
                    image = file.read()
            elif task_code == 35:
                # file_list = glob.glob(fr'user_file/{user_id}/WaveletPacket.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/WaveletPacket.png", 'rb') as file:
                    image = file.read()
            elif task_code == 36:
                # file_list = glob.glob(fr'user_file/{user_id}/FastKurtogram_filtered_data.*')
                # with open(file_list[0], 'rb') as file:FastKurtogram_filtered_data.png
                with open(f"user_file/{user_id}/Fast_kurtogram.png", 'rb') as file:
                    image = file.read()
            elif task_code == 37:
                # file_list = glob.glob(fr'user_file/{user_id}/PSD_filtered_data.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/PSD_filtered_data.png", 'rb') as file:
                    image = file.read()
            elif task_code == 41:
                # file_list = glob.glob(fr'user_file/{user_id}/chrome插件位置.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/chrome插件位置.png", 'rb') as file:
                    image = file.read()
            elif task_code == 42:
                # file_list = glob.glob(fr'user_file/{user_id}/feature_selection.*')
                # with open(file_list[0], 'rb') as file:
                with open(f"user_file/{user_id}/feature_selection.png", 'rb') as file:
                    image = file.read()
            elif task_code == 51:
                # file_list = glob.glob(fr'user_file/{user_id}/confusion_matrix.*')
                # with open(file_list[0], 'rb') as file:
                try:
                    with open(f"user_file/{user_id}/confusion_matrix.png", 'rb') as file:
                        image = file.read()
                except:
                    try:
                        with open(f"user_file/{user_id}/rul_figure.png", 'rb') as file:
                            image = file.read()
                    except:
                        with open(f"user_file/{user_id}/optimization_curve.png", 'rb') as file:
                            image = file.read()
        except:
            return {"message": "用户还未提交任务"}, 401

        # image = open(f"user_file/{user_id}/signal_based_bearing_defect_model_inner.png", 'rb').read()
        resp = Response(image, mimetype="image/png")
        return resp



