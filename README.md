MaxEnt
======

这是一个最大熵的简明Java实现，提供提供训练与预测接口。训练算法采用GIS训练算法，附带示例训练集和一个天气预测的Demo。

MaxEnt训练与预测
--

 - 调用方法
 
```java
public static void main(String[] args) throws IOException
{
        String path = "data/train.txt";
        MaxEnt maxEnt = new MaxEnt();
        maxEnt.loadData(path);
        maxEnt.train(200);
        List<String> fieldList = new ArrayList<String>();
        fieldList.add("Sunny"); // 假如天晴
        fieldList.add("Humid"); // 并且湿润
        Pair<String, Double>[] result = maxEnt.predict(fieldList);  // 预测出门和自宅的概率各是多少
        System.out.println(Arrays.toString(result));
}
```
 - 算法详解
 最大熵属于辨识模型,能够满足所有已知的约束, 对未知的信息不做任何过分的假设。
 详见[《最大熵的Java实现》][1]

  [1]: http://www.hankcs.com/nlp/maximum-entropy-java-implementation.html
