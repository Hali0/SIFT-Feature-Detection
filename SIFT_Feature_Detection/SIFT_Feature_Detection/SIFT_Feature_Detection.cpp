//功能：SIFT特征点提取并进行匹配
//版本：version1.0.0
//所用opencv版本: opencv2.4.9
//备注：目前仅能在release下进行测试，debug下运行有时会出现内存冲突

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<vector>

using namespace std;
using namespace cv;

int main()
{
	//从文件中读入原图像
	const char* imagename1 = "img/box.png";
	const char* imagename2 = "img/box_in_scene.png";
	Mat img = imread(imagename1);
	Mat img2 = imread(imagename2);

	//检测图像是否读入成功
	if (img.empty())
	{
		cout << "Can not load the image " << imagename1 << endl;
		return -1;
	}
	if (img2.empty())
	{
		cout << "Can not load the image " << imagename2 << endl;
		return -1;
	}

	//显示图像
	imshow("image before", img);
	imshow("image2 before", img2);


	//sift特征检测
	SiftFeatureDetector  siftdtc;
	vector<KeyPoint>kp1, kp2;

	//提取特征点的具体位置和角度，其保存在kp中
	siftdtc.detect(img, kp1);
	Mat outimg1;
	drawKeypoints(img, kp1, outimg1);
	imshow("image1 keypoints", outimg1);
	KeyPoint kp;

	vector<KeyPoint>::iterator itvc;
	//显示特征点位置及角度
	for (itvc = kp1.begin();itvc != kp1.end();itvc++)
	{
		cout << "angle:" << itvc->angle << "\t" << itvc->class_id << "\t" << itvc->octave << "\t" << itvc->pt << "\t" << itvc->response << endl;
	}

	siftdtc.detect(img2, kp2);
	Mat outimg2;
	drawKeypoints(img2, kp2, outimg2);
	imshow("image2 keypoints", outimg2);

	int64 t = getTickCount();

	//特征提取器
	SiftDescriptorExtractor extractor;
	//保存实验要用的特征  
	Mat descriptor1, descriptor2;
	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;
	Mat img_matches;
	extractor.compute(img, kp1, descriptor1);
	extractor.compute(img2, kp2, descriptor2);

	//显示特征提取所用时间
	t = getTickCount() - t;
	cout << "Time elapsed: " << t * 1000 / getTickFrequency() << "ms" << endl;

	//imshow("desc", descriptor1);
	//cout << endl << descriptor1 << endl;
	matcher.match(descriptor1, descriptor2, matches);
	
	//显示匹配后的图像
	drawMatches(img, kp1, img2, kp2, matches, img_matches);
	imshow("matches", img_matches);

	//等待按键
	waitKey();
	return 0;
}