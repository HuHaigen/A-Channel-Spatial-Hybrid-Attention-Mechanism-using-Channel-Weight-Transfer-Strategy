"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 工具库
"""
import tensorflow as tf
def choose_nets(nets_name, num_classes=102):
    #nets_name = nets_name.lower()
    ## VGG
    if nets_name == 'vgg11':
        from models.VGG import VGG11
        return VGG11(num_classes)
    if nets_name == 'vgg13':
        from models.VGG import VGG13
        return VGG13(num_classes)
    if nets_name == 'vgg16':
        from models.VGG import VGG16
        return VGG16(num_classes)
    if nets_name == 'vgg19':
        from models.VGG import VGG19
        return VGG19(num_classes)
    ## ResNet
    if nets_name == 'ResNet18':
        from models.ResNet import ResNet18
        return ResNet18(num_classes)
    if nets_name == 'ResNet34':
        from models.ResNet import ResNet34
        return ResNet34(num_classes)
    if nets_name == 'ResNet50':
        from models.ResNet import ResNet50
        return ResNet50(num_classes)
    if nets_name == 'ResNet101':
        from models.ResNet import ResNet101
        return ResNet101(num_classes)
    if nets_name == 'ResNet152':
        from models.ResNet import ResNet152
        return ResNet152(num_classes)

    # ResNeXt
    if nets_name == 'ResNeXt50':
        from models.resnext import ResNeXt50
        return ResNeXt50(num_classes)

    #MobileNetV2
    if nets_name == 'MobileNetV2':
        from models.MobileNetV2 import MobileNetV2
        return MobileNetV2(num_classes)

    # GoogleNet
    if nets_name == 'GoogLeNet':
        from models.GoogLeNet import GoogLeNet
        return GoogLeNet(num_classes)
    # DenseNet
    if nets_name == 'DenseNet161':
        from models.DenseNet import densenet161
        return densenet161(num_classes)

    raise NotImplementedError

def build_optimizer(learning_rate=0.1, momentum=0.9):
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #[500, 32000, 48000],
        #[1000, 64000, 96000],
        [12000,24000,32000,40000,48000],
        [learning_rate, 0.02, 0.004,0.0008,0.0003,0.0001])
        #[learning_rate / 10., learning_rate, learning_rate / 10., learning_rate / 100.])

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    return optimizer

def generate_desc_csv(root_folder):
    """
    根据数据集路径生成数据集说明的csv文件
    :param root_folder:
    :return:
    """
    import os
    from tqdm import tqdm
    import pandas as pd
    from glob import glob
    # 字符编码为数值
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root_folder))):
        if not os.path.isdir(os.path.join(root_folder, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())
    file_id = []
    label = []
    for category in tqdm(os.listdir(root_folder)):
        images = glob(os.path.join(root_folder, category)+'/*')
        for img in images:
            file_id.append(img.replace("\\", "/"))  # 契合linux平台
            label.append(name2label[category])

    df_desc = pd.DataFrame({'file_id': file_id, 'label': label})
    df_desc.to_csv("../data/desc.csv", encoding="utf8", index=False)


def save_pickle(python_object, saved_path):
    """
    保存Python对象为pickle文件
    :param python_object:
    :param saved_path:
    :return:
    """
    import pickle
    output = open(saved_path, 'wb')
    pickle.dump(python_object, output)
    output.close()


def load_pickle(file_path):
    """
    加载本地的pickle文件为Python对象
    :param file_path:
    :return:
    """
    import pickle
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


if __name__ == '__main__':
    generate_desc_csv("../data/Caltech101")


