from moduls.data_augmentation.GAN.GAN_main import GAN_all_main
from moduls.data_augmentation.image_transformation_DA.image_transformation_DA_main import image_transformation_DA_main
from moduls.data_augmentation.Report_data_augmentation import *


from moduls.data_augmentation.Monte_Carlo_sampling.Monte_Carlo_main import *


def data_augmentation_main(data,label=None,switch=1, GAN_num=2, Z_dim=100, image_transform_multi=2, deltax=(-50, 0, 50),
                           deltay=(-0.2, 0.2), rot=(-0.01, -0.015, 0, 0.01, 0.015), snr=(20, 15, 10),
                           rescale=(1 / 3, 1 / 2, 2, 3), image_transformation_DA_switch=(1, 1, 1, 1),
                           mode=0,distribution=0,function_select=0,m=100,a=0.05,b=0.003,c=0,d=0.003,e=-0.001,
                           save_path='./',output_file=0,output_image=0):
    '''
    :param data:input signal，一维数组,(10001,)
    :param switch: to determine which augmentation method. int, e.g. 1,2,3....
    :param GAN_num: number of augmentatation in GAN
    :param multi: number of augmentatation in image_transformation_DA
    :param save_path:path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    if switch == 2:
        newdata_all,newlabel_all = GAN_all_main(data,label,GAN_num,Z_dim,save_path,output_file,output_image)

        # 生成报告
        word_GAN(data, label, newdata_all, newlabel_all, GAN_num, Z_dim, save_path, output_file, output_image)
        return newdata_all, newlabel_all
    elif switch == 1:

        newdata_all, newlabel_all = image_transformation_DA_main(data, label,image_transform_multi, deltax,deltay,rot,snr,
                                                    rescale, image_transformation_DA_switch,save_path,
                                                    output_file, output_image)

        # 生成报告
        word_image_transformation(data, label, image_transform_multi, newdata_all, newlabel_all, deltax, deltay, rot, snr, rescale,
                                   save_path, output_file, output_image)
        return newdata_all, newlabel_all

    elif switch == 3:#当选用此种方法时，使用Monte_writein函数读取文件，输入数据data也和另两种方法不一样，应当区分
        newdata_all = Monte_Carlo_DA(data, save_path, mode, distribution, function_select,m,a,b,c,d,e,output_file,output_image)

        # 生成报告
        word_Monte_Carlo(data, newdata_all, mode, distribution, function_select, m, a, b, c, d, e, save_path, output_file,
                         output_image)

        return newdata_all
if __name__=="__main__":
    pass
    # loadpath = writein('1.mat',1)
    # # print(loadpath.shape)
    # data = loadpath[:3,:]
    # # data='theta.mat'
    # # # data="./folder"
    # augmentation_data=data_augmentation_main(data,switch=2, mode=1,output_file=0)

