from datetime import datetime

from app import db


class User(db.Model):
    """
    应乎基本信息表
    """
    # TODO: user功能完善"user_base"
    __tablename__ = "user"

    class GENDER:
        MAN = 0
        WOMAN = 1
        NEUTRAL = 2

    # 字段
    user_id = db.Column(db.Integer, primary_key=True, doc="用户id")
    user_account = db.Column(db.String(40), doc="用户账号")
    user_password = db.Column(db.String(30), doc="用户密码")
    user_name = db.Column(db.String(30), doc="用户姓名")
    user_gender = db.Column(db.Integer, doc="性别")
    user_birthday = db.Column(db.DateTime, doc="生日")
    user_introduction = db.Column(db.String(50), doc="简介")
    last_login = db.Column(db.DateTime, doc="最后登陆时间")
    vip = db.Column(db.Integer, doc="会员状态")
    vip_deadline = db.Column(db.DateTime, doc="会员截止日期")

    def to_dict(self):
        """模型转字典"""

        return {
            'user_id': self.user_id,
            'user_account': self.user_account,
            'user_password': self.user_password,
            'user_name': self.user_name,
            'user_gender': self.user_gender,
            'user_birthday': self.user_birthday,
            'user_introduction': self.user_introduction if self.user_introduction else "",
            'last_login': self.last_login,
            'vip': self.vip,
            'vip_deadline': self.vip_deadline
        }


class UserParameter(db.Model):
    """用户提交工具箱配置参数保存"""

    __tablename__ = "user_parameter"

    id = db.Column('legalize_id', db.Integer, primary_key=True, doc='工具箱配置参数id')
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), doc='用户id')

    # 模块1-5功能选择参数: 0代表未选择该模块
    bearing_simulation = db.Column(db.Integer, doc='modul_1: bearing_simulation')
    data_augmentation = db.Column(db.Integer, doc='modul_2: data_augmentation')
    data_preprocessing = db.Column(db.Integer, doc='modul_3: data_preprocessing')
    features_extraction = db.Column(db.Integer, doc='modul_4: features_extraction')
    fault_diagnose = db.Column(db.Integer, doc='modul_5: fault_diagnose')

    # 模块1:相关参数
    Do_phy = db.Column(db.Float, doc='[mm]')
    Di_phy = db.Column(db.Float, doc='[mm]')
    dd_phy = db.Column(db.Float, doc='[mm]')
    D_phy = db.Column(db.Float, doc='[mm]')
    Kb_phy = db.Column(db.Float, doc='[N/mm]')
    alpha_phy = db.Column(db.Float, doc='[deg]')
    Nb = db.Column(db.Float, doc='[None]')
    Ms = db.Column(db.Float, doc='[kg]')
    Mp = db.Column(db.Float, doc='[kg]')
    Mr = db.Column(db.Float, doc='[kg]')
    Ks = db.Column(db.Float, doc='[N/mm]')
    Kr = db.Column(db.Float, doc='[N/mm]')
    Cs = db.Column(db.Float, doc='[Ns/mm]')
    Cp = db.Column(db.Float, doc='[Ns/mm]')
    Cr = db.Column(db.Float, doc='[Ns/mm]')
    L = db.Column(db.Float, doc='[mm]')
    B_phy = db.Column(db.Float, doc='[mm]')
    H = db.Column(db.Float, doc='[mm]')
    ORS = db.Column(db.Integer, doc='Outer ring switch')
    IRS = db.Column(db.Integer, doc='Inner ring switch')
    BS = db.Column(db.Integer, doc='Ball ring switch')
    ORN = db.Column(db.Integer, doc='Outer ring number')
    IRN = db.Column(db.Integer, doc='Inner ring number')
    BN = db.Column(db.Integer, doc='Ball ring number')
    Fr = db.Column(db.Float, doc='[N]')
    Fa = db.Column(db.Float, doc='[N]')
    Omega_shaft = db.Column(db.Float, doc='[Hz]')
    max_step = db.Column(db.Float, doc='[None]')
    step_size_phy = db.Column(db.Float, doc='[s]')
    duration_phy = db.Column(db.Float, doc='[s]')
    mutation_percentage = db.Column(db.Float, doc='[None]')
    initial_angular_position = db.Column(db.Float, doc='[None]')
    Do_sig = db.Column(db.Float, doc='[mm]')
    Di_sig = db.Column(db.Float, doc='[mm]')
    z_sig = db.Column(db.Integer, doc='[None]')
    D_sig = db.Column(db.Float, doc='[mm]')
    type_factor_sig = db.Column(db.Float, doc='[None]')
    alpha_sig = db.Column(db.Float, doc='[deg]')
    load_max = db.Column(db.Float, doc='[N]')
    shaft_speed = db.Column(db.Float, doc='[Hz]')
    phi_limit = db.Column(db.Float, doc='[deg]')
    load_proportional_factor = db.Column(db.Float, doc='[None]')
    resonance_frequency = db.Column(db.Float, doc='[Hz]')
    load_distribution_parameter = db.Column(db.Float, doc='[None]')
    defect_type = db.Column(db.Integer, doc='[None]')
    defect_initial_position = db.Column(db.Float, doc='[deg]')
    B_sig = db.Column(db.Float, doc='[None]')
    step_size_sig = db.Column(db.Float, doc='[s]')
    duration_sig = db.Column(db.Float, doc='[s]')

    # 模块2:相关参数
    translation = db.Column(db.Integer, doc='[None]数据增强—平移（0代表未选择）')
    rotation = db.Column(db.Integer, doc='[None]数据增强—旋转（0代表未选择）')
    noise = db.Column(db.Integer, doc='[None]数据增强—噪音（0代表未选择）')
    scale = db.Column(db.Integer, doc='[None]数据增强—缩放（0代表未选择）')
    data_structure = db.Column(db.Integer, doc='[None]0:竖直（vertically）&1:水平（horizontally）')
    training_samples = db.Column(db.Integer, doc='[None]0:Noise Data&1:Input Data（horizontally）')
    data_num = db.Column(db.Integer, doc='[None]生成样本个数')

    # 模块3:相关参数
    denoise_method = db.Column(db.Integer, doc='降噪方法：0:小波 &1:小波包 &2:FFT')
    class_of_filter = db.Column(db.Integer, doc='0:Sample Filter &1:Kurtosis Filter &2:Frequency Band Filter &3:Frequency Band to be Filter')
    wavelet_sf = db.Column(db.Integer, doc='Sample Filter:波选择')
    level_sf = db.Column(db.Float, doc='[None]')
    wavelet_kf = db.Column(db.Integer, doc='Kurtosis Filter:波选择')
    level_kf = db.Column(db.Float, doc='[None]')
    del_num = db.Column(db.Integer, doc='[None] Del Num of Kurtosis')
    wavelet_fbf = db.Column(db.Integer, doc='Frequency Band Filter:波选择')
    max_frequency = db.Column(db.Float, doc='[None]生成样本个数')
    max_fb = db.Column(db.Float, doc='[Hz] Frequency Band to be Filter: MAX')
    min_fb = db.Column(db.Float, doc='[Hz] Frequency Band to be Filter: MIN')
    filter_fft = db.Column(db.Integer, doc='FFT Filter 选择')
    min_fft = db.Column(db.Float, doc='[Hz] FFT: MIN')
    max_fft = db.Column(db.Float, doc='[Hz] FFT: MAX')
    sampling_rate_fft = db.Column(db.Integer, doc='[Hz]')

    # 模块4:相关参数
    t_features = db.Column(db.Integer, doc='[None]')
    wavelet_f_features = db.Column(db.Integer, doc='[None]')
    wavelet_packet_f_features = db.Column(db.Integer, doc='[None]')
    envelope_f_features = db.Column(db.Integer, doc='[None]')
    stf_features = db.Column(db.Integer, doc='[None]')
    class_of_wavelet = db.Column(db.Integer, doc='[None]0:Wavelet Transform &1:Wavelet Packet Transform')
    wavelet_ff = db.Column(db.Integer, doc='[None]小波（包）特征提取:波选择')
    level_ff = db.Column(db.Float, doc='[None]')
    sampling_rate_env = db.Column(db.Float, doc='[Hz]')
    ball_diameter = db.Column(db.Float, doc='[mm]')
    pitch_diameter = db.Column(db.Float, doc='[mm]')
    contact_angel = db.Column(db.Float, doc='[deg]')
    number_balls = db.Column(db.Float, doc='[None]')
    rotating_speed = db.Column(db.Float, doc='[r/s]')
    feature_selection = db.Column(db.Integer, doc='[r/s]')

    # 模块5:相关参数
    ml_method = db.Column(db.Integer, doc='[None]机器学习方法：0:KNN &1:SVM &2:DT &3:RF &4:CNN &5:LSTM')
    opt_algorithm = db.Column(db.Integer, doc='[None]优化算法：0:None &1:PSO &2:GA &3:SA')
    k_knn = db.Column(db.Integer, doc='[None]KNN参数')
    w_knn = db.Column(db.Integer, doc='[None]KNN参数')

    # 其他参数
    download_status = db.Column(db.Boolean, default=False, doc="下载链接状态")
    # ctime = db.Column('create_time', db.DateTime, default=datetime.now, doc='创建时间')
    # utime = db.Column('update_time', db.DateTime, default=datetime.now, doc='更新时间')

    def to_dict(self):
        """模型转字典"""

        return {
            'id': self.id,
            'user_id': self.user_id,
            'modul_1': self.modul_1,
            'modul_2': self.modul_2,
            'parm1': self.parm1,
            'download_url': self.download_url,
            'ctime': self.ctime,
            'utime': self.utime,
        }

