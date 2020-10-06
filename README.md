# 作业说明

## Q1
第一个作业有两种版本 你可以使用 jupyter notebook 或是 Gradle project. 选择一个作就行
#### Gradle Project
1. `ImageClassification.java` 里面有事先实现好部分的图片分类，少了模型读取的部分，请完成87行的代码读取项目内的`tracedd_resnet18.pt`的模型
2. 代码里会读去一张萨摩耶的图片，但是目前的分类结果并不是萨摩耶，因为数据前处理的代码有问题，请把第77行的`reshape` 移除并且实现 toTensor()
* 注意不能直接调用 NDImageUtils.toTensor() 
tips:
* toTensor() 方法做了三件事 
    1. 把 NDArray 里面每个元素从0到255归一化到0到1
    2. 把 NDArray shape 从 (Height, Width, Channel) 转成 (Channel, Height, Width)
    3. 把数据形态从 uint8 转成 float32
    原来的实现方法是用了`reshape`但底下的数据没有发生任何改变，我们需要用一个会将底下数据排列方式改变的算子．
3. 完成任务 2 之后会发现分类结果正确显示出 Samoyed 但标签后面概率的值是错的，因为在`processOutput()`里面没有加上 softmax()，请实现 softmax算子
* 注意不能直接调用 array.softmax()
 
#### Jupyter Notebook
在使用 jupyter notebook 请先完成 [Setup](https://github.com/awslabs/djl/blob/master/jupyter/README.md#setup)
1. 照着 jupyter notebook 上的说明完成 notebook
 
## Q2 (加分任务)
将 DJL 与其他基于 Java 的框架结合完成一个任务
1. 模型选择：可以选用任何模型，来自DJL模型库或者其他地方。
2. 结合思路：可以和Spark，Flink，Beam等基于Java的框架集成，可以参考 [Apache Flink](https://github.com/aws-samples/djl-demo/tree/master/flink/sentiment-analysis), [Apache Beam](https://github.com/aws-samples/djl-demo/tree/master/apache-beam/ctr-prediction), [Apache Spark](https://github.com/aws-samples/djl-demo/tree/master/spark/spark3.0/image-classification)