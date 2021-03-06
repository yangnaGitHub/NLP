目录和重要文件说明:
 data:数据以及数据处理相关
 Embed:词embedding相关,自己的embedding
 LSTM:RNN,LSTM等相关模型
 Normal_DL:normal深度学习相关
 procedure:一部分实验记录
 Test:测试文件
 TextCNN:cnn,textcnn模型相关
 TFIDF:tfidf模型相关
 visio:各个模型相关的visio图
 Allmodule.cfg:总配置文件
 main.py:启动文件
 some_note.txt:备忘

代码说明:
 1>启动函数是main.py
  启动不用带任何参数,具体的关于模型的配置在Allmodule.cfg文件中
  修改的方法见文件中具体的说明,如果修改规则不满足你的要求,请修改conf.py中的load规则
 2>每个模型一个文件夹
  模型文件中一般存在4个文件,模型的配置是module.cfg
  如果修改配置文件有问题的话要出错请确认module.py
  opmodule.py一般是和主目录下的Allmodule.py对接的文件
  readme.txt是模型的说明,使用的什么,注意事项等
  model_record一般是放各个模型版本的模型简图和配置文件以及模型代码和模型
 3>一般的数据文件和处理都是放在data目录下
 4>做过的验证都放在procedure目录下
  文件夹的命名:第几次+准确率
  文件夹下面一般包含整个模型有哪些修改单个模型使用的配置文件以及配置文件的详细说明
  修改的内容以及结果说明
 5>model一般放的是各个模型的保存下来的模型以及log
  每次模型有改动的时候或是模型有保存的需要的话建议您修改各个模型目录下面的module.cfg(这是默认的文件名,文件的名称请参照Allmodule.cfg中的use_model下的modele_conf字段,无定义使用默认字段)中的module_path
  module_path的命名规则化
   ./model/TextCNN/TextCNN01_001/model ==> 使用的是TextCNN下的模型(model_record)TextCNN01模型和其下的module_001.cfg配置文件
 6>train_info.txt是训练的地址
  使用SSH协议连接上给出的Host就可以训练

使用说明:
 1>修改模型一般是修改Allmodule.cfg以及每个模型目录下的module.cfg,注意出错的话可能要同步修改module.py
 2>重要的事情说3词:各个配置文件很重要,修改的时候要注意遵守规则或是修改加载规则