from flask_sqlalchemy import SQLAlchemy, SignallingSession, get_state
from sqlalchemy import orm
import random

# from .session import RoutingSession

# 1. 自定义Ses
# sion类,继承SignallingSession, 并重写get_bind方法
from sqlalchemy import orm

class RoutingSession(SignallingSession):
    def __init__(self, *args, **kwargs):
        super(RoutingSession, self).__init__(*args, **kwargs)
        # 随即选择从数据库的key
        self.slave_key = random.choice(["slave1", "slave2"])

    def get_bind(self, mapper=None, clause=None):
        """每次数据库操作(增删改查/事物操作)都会调用该方法,来获取对应的数据库引擎"""
        state = get_state(self.app)

        # 按照模型中制定的数据库,返回数据库引擎
        if mapper is not None:
            try:
                # SA >= 1.3
                persist_selectable = mapper.persist_selectable
            except AttributeError:
                # SA <=1.3
                persist_selectable = mapper.mapped_table
            # 如果项目中指明了特定数据库,就获得bind_key指明的数据库,进行数据库绑定
            info = getattr(persist_selectable, 'info', {})
            bind_key = info.get('bind_key')
            if bind_key is not None:
                return state.db.get_engine(self.app, bind=bind_key)

        from sqlalchemy.sql.dml import UpdateBase
        if self._flushing or isinstance(clause, UpdateBase):
            # 写操作--主数据库
            print("写操作--主数据库")
            return state.db.get_engine(self.app, bind="master")
        else:
            # 读操作--从数据库
            print("读操作--从数据库")
            return state.db.get_engine(self.app, bind=self.slave_key)

class RoutingSQLAlchemy(SQLAlchemy):
    """
    2. 自定义SQLALchemy类,重写create_session方法
    """

    def create_session(self, options):
        return orm.sessionmaker(class_=RoutingSession, db=self, **options)
