#include "KalmanTracker.h"

int KalmanTracker::kf_count = 0;

// 	初始化卡尔曼滤波器
void KalmanTracker::init_kf(StateType stateMat,int ClassId)
{
	int stateNum = 7;			//使用7个状态值，把IOU的框也给加进去了
	int measureNum = 4;
	kf = KalmanFilter(stateNum, measureNum, 0);

	classId = ClassId;

	measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum)<<			//状态转移矩阵
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);
    
   setIdentity(kf.measurementMatrix);									//观测矩阵
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));					//过程噪声协方差矩阵
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));				//测量噪声协方差矩阵
	setIdentity(kf.errorCovPost, Scalar::all(1));						//估计值和真实值之间的误差协方差矩阵
	
	//使用[cx,cy,s,r]格式的标定框来初始化状态向量
  kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}

//预测估计的标定框
StateType KalmanTracker::predict()
{
	// predict
	Mat p = kf.predict();
	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	m_history.push_back(predictBox);
	return m_history.back();
}

//使用观测的标定框来更新状态向量
void KalmanTracker::update(StateType stateMat, int ClassId)
{
	m_time_since_update = 0;
	m_history.clear();
	m_hits += 1;
	m_hit_streak += 1;
	classId = ClassId;
	// measurement
	measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	measurement.at<float>(1, 0) = stateMat.y + stateMat.heigh / 2;
	measurement.at<float>(2, 0) = stateMat.area();
	measurement.at<float>(3, 0) = stateMat.width /  stateMat.height;

	// update
	kf.correct(measurement);
}

//返回当前状态向量
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}

//把标定框从[cx,cy,s,r]格式转成[x,y,w,h]
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}



