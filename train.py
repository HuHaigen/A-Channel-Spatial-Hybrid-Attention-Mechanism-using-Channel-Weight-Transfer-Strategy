"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 模型训练
"""
import tensorflow as tf
from tensorflow import keras
#from models import vgg16, resnet50, densenet121
from my_scripts.data import load_data
from my_scripts.utils import choose_nets, build_optimizer
from my_scripts.visualize import plot_history
from evaluation import*
from draw import*
from to_excel import*
import numpy as np

print(tf.test.is_gpu_available())

def load_db(batch_size):
    """
    加载数据集
    :param batch_size:
    :return:
    """
    db_train, db_test = load_data("./data/desc.csv", batch_size)
    return db_train, db_test


if __name__ == '__main__':


    #vgg = vgg16((None, 224, 224, 3), 102)
    #resnet = resnet50((None, 224, 224, 3), 102)
    #densenet = densenet121((None, 224, 224, 3), 102)

    training_epochs = 300
    batchsize = 16
    learning_rate = 0.1
    momentum = 0.9

    Attention_name = "se"
    model = choose_nets('ResNet34')
    model_name = 'Caltech101_ResNet34_se_models.h5'
    logdir = './callbacks/Caltech101_callbacks_ResNet34_se'
    drawdir = './Results/ResNet'
    drawname = "cifar100_ResNet34_se"

    dir_lossTrain = './Results/ResNet/Caltech101_ResNet34_loss_train.xlsx'
    dir_lossVal = './Results/ResNet/Caltech101_ResNet34_loss_val.xlsx'
    dir_evalutionTrain = './Results/ResNet/Caltech101_ResNet34_evolution_train.xlsx'
    dir_evalutionVal = './Results/ResNet/Caltech101_ResNet34_evolution_val.xlsx'
    """" 

        Attention_name = "my"
    model = choose_nets('ResNet50')
    model_name = 'Caltech101_ResNet18_eca_models.h5'
    logdir = './callbacks/Caltech101_callbacks_ResNet18_eca'
    drawdir = './Results/ResNet'
    drawname = "cifar100_ResNet18_eca"

    dir_lossTrain = './Results/ResNet/Caltech101_ResNet18_loss_train.xlsx'
    dir_lossVal = './Results/ResNet/Caltech101_ResNet18_loss_val.xlsx'
    dir_evalutionTrain = './Results/ResNet/Caltech101_ResNet18_evolution_train.xlsx'
    dir_evalutionVal = './Results/ResNet/Caltech101_ResNet18_evolution_val.xlsx'
    
    
    Attention_name = "none"
    model = choose_nets('ResNet50')
    model_name = 'Caltech101_ResNet50_none_models.h5'
    logdir = './callbacks/Caltech101_callbacks_ResNet50_none'
    drawdir = './Results/ResNet'
    drawname = "cifar100_ResNet50_none"

    dir_lossTrain = './Results/ResNet/Caltech101_ResNet50_loss_train.xlsx'
    dir_lossVal = './Results/ResNet/Caltech101_ResNet50_loss_val.xlsx'
    dir_evalutionTrain = './Results/ResNet/Caltech101_ResNet50_evolution_train.xlsx'
    dir_evalutionVal = './Results/ResNet/Caltech101_ResNet50_evolution_val.xlsx'
    
  

    Attention_name = "none"
    model = choose_nets('MobileNetV2')
    model_name = 'Caltech101_MobileNetV2_none_models.h5'
    logdir = './callbacks/Caltech101_callbacks_MobileNetV2_none'
    drawdir = './Results/ResNet'
    drawname = "Caltech101_MobileNetV2_none"

    dir_lossTrain = './Results/ResNet/Caltech101_MobileNetV2_loss_train.xlsx'
    dir_lossVal = './Results/ResNet/Caltech101_MobileNetV2_loss_val.xlsx'
    dir_evalutionTrain = './Results/ResNet/Caltech101_MobileNetV2_evolution_train.xlsx'
    dir_evalutionVal = './Results/ResNet/Caltech101_MobileNetV2_evolution_val.xlsx'

    """

    #optimizers = tf.keras.optimizers.Adam(1e-4)
    optimizers = build_optimizer(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizers,
                  loss="categorical_crossentropy",
                  metrics=['accuracy',top_1_accuracy, top_3_accuracy, precision, recall, macro_f1])

    train_dataset, val_dataset = load_db(batchsize)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    output_model_file = os.path.join(logdir, model_name)
    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        # keras.callbacks.ModelCheckpoint(output_model_file,
        #                               save_best_only=True),  # 只保存最好的模型
        # keras.callbacks.EarlyStopping(patience=50, min_delta=1e-3)
    ]

    history = model.fit(train_dataset,
                        epochs=training_epochs,
                        validation_data=val_dataset)
 # draw.py
    plot_learning_curves(history, drawdir, drawname)
    #plot_learning_curves(history)
   # plot_history(history)

    ## 提取数据
    train_loss = history.history['loss']
    train_top1 = history.history['top_1_accuracy']
    train_top3 = history.history['top_3_accuracy']
    train_pre = history.history['precision']
    train_rec = history.history['recall']
    #train_kappa = history.history['kappa_score']
    train_F1 = history.history['macro_f1']


    val_loss = history.history['val_loss']
    val_top1 = history.history['val_top_1_accuracy']
    val_top3 = history.history['val_top_3_accuracy']
    val_pre = history.history['val_precision']
    val_rec = history.history['val_recall']
    val_F1 = history.history['val_macro_f1']

    ## 保存数据


    train_loss2excel(training_epochs, train_loss, Attention_name,dir_lossTrain)
    val_loss2excel(training_epochs, val_loss, Attention_name, dir_lossVal)

    eval_to_excel(training_epochs, train_top1, train_top3, train_pre, train_rec, train_F1, Attention_name, dir_evalutionTrain)
    eval_to_excel(training_epochs, val_top1, val_top3, val_pre, val_rec, val_F1, Attention_name, dir_evalutionVal)


    ## 模型参数量
    #model.summary()
    COUNTS = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print("parameters / M:", float(COUNTS)/1024/1024)

   # model.save(os.path.join(logdir, model_name))

    ## FLOPS

    #flops = get_flops(model, batch_size=1)
    #print(f"FLOPS: {flops / 10 ** 9:.03} G")


