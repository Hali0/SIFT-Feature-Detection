//���ܣ�SIFT��������ȡ������ƥ��
//�汾��version1.0.0
//����opencv�汾: opencv2.4.9
//��ע��Ŀǰ������release�½��в��ԣ�debug��������ʱ������ڴ��ͻ

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<vector>

using namespace std;
using namespace cv;

int main()
{
	//���ļ��ж���ԭͼ��
	const char* imagename1 = "img/box.png";
	const char* imagename2 = "img/box_in_scene.png";
	Mat img = imread(imagename1);
	Mat img2 = imread(imagename2);

	//���ͼ���Ƿ����ɹ�
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

	//��ʾͼ��
	imshow("image before", img);
	imshow("image2 before", img2);


	//sift�������
	SiftFeatureDetector  siftdtc;
	vector<KeyPoint>kp1, kp2;

	//��ȡ������ľ���λ�úͽǶȣ��䱣����kp��
	siftdtc.detect(img, kp1);
	Mat outimg1;
	drawKeypoints(img, kp1, outimg1);
	imshow("image1 keypoints", outimg1);
	KeyPoint kp;

	vector<KeyPoint>::iterator itvc;
	//��ʾ������λ�ü��Ƕ�
	for (itvc = kp1.begin();itvc != kp1.end();itvc++)
	{
		cout << "angle:" << itvc->angle << "\t" << itvc->class_id << "\t" << itvc->octave << "\t" << itvc->pt << "\t" << itvc->response << endl;
	}

	siftdtc.detect(img2, kp2);
	Mat outimg2;
	drawKeypoints(img2, kp2, outimg2);
	imshow("image2 keypoints", outimg2);

	int64 t = getTickCount();

	//������ȡ��
	SiftDescriptorExtractor extractor;
	//����ʵ��Ҫ�õ�����  
	Mat descriptor1, descriptor2;
	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;
	Mat img_matches;
	extractor.compute(img, kp1, descriptor1);
	extractor.compute(img2, kp2, descriptor2);

	//��ʾ������ȡ����ʱ��
	t = getTickCount() - t;
	cout << "Time elapsed: " << t * 1000 / getTickFrequency() << "ms" << endl;

	//imshow("desc", descriptor1);
	//cout << endl << descriptor1 << endl;
	matcher.match(descriptor1, descriptor2, matches);
	
	//��ʾƥ����ͼ��
	drawMatches(img, kp1, img2, kp2, matches, img_matches);
	imshow("matches", img_matches);

	//�ȴ�����
	waitKey();
	return 0;
}