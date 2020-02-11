

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)


### Bert plus LSTM
#### 一个简单的NLP项目（文本情感分析）的Bert baseline ，flask后端API，修改了全局model load的方式，增加了模型推理的速度，使用nginx搭配Gunicorn启动Flask，使用虚拟环境搭配sh的启动方式，可以直接对model进行一键重启，并有错误日志监控，使用post请求，url= 'http://127.0.0.1:5000/sentiment_analysis_api'



#### 下载预训练的Bert模型， bert_local :  D:/bert文本分类/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12

> bert可视化架构说明  [click](http://jalammar.github.io/illustrated-transformer/)
 
## 输出结果
> 第一次使用初始化的时候比较耗时，第二次预测的速度明显加快，后面的推理就正常了

```

{'content': '这家酒店很垃圾', 'sa': '0.181633.4'}

time used:0.08649280000000001

```


