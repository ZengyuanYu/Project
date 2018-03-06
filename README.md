### Classify_plants
作物识别与分类（PyTorch）
### add_hat
给人物加上圣诞帽
### words_frequency
词频统计
- 开发环境
	- Anaconda3.6.3 + Pycharm2017.3.2
	- collections json re sklearn安装升级至高版本 conda update --all

- 代码实现
	- data_clean()实现数据清洗并转换成为列表
	- create_vocal_list()去除重复单词
	- words_frequency() 词频统计

- 一百万条文本数据
单文件大数据量的读取，全部读至内存会引发内存溢出错误，使用
生成器逐行读取文本信息来解决大文件读取出错的问题，至于加速，
多文件的读取可以使用多进程的办法来处理，而大数据单文件可能
会使用Spark，涉及知识盲区，回答可能拿不住要点。

- 训练
使用sklearn包
	- 特征抽取
	首先将text string逐行读取到内存中，然后进行数据清洗成有效
	数据并拼接每一行，去重之后使用词袋模型将数据转换成向量
	- 模型选择
	选用机器学习方法中的SVM
	- 评估
	使用3折交叉验证，即共三种排列组合方式进行训练和验证，得到三个结果
