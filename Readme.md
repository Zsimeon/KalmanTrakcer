1 使用卡尔曼滤波进行多目标追踪

2 给定一个道路视频，并通过检测算法得到车道中所有的车的boundingBox信息

3 追踪多个车框，每个车框一个车辆id，并将表示同一车辆的车框赋予相同的id

4 使用visual studio 2019 和 opencv 4.1.0

5 使用匈牙利匹配将预测框和实际检测到的框相匹配

项目中涉及的卡尔曼滤波和匈牙利匹配方法来自该作者的github
https://github.com/mcximing/sort-cpp
