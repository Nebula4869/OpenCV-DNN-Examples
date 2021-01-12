#include "YOLOv3.h"
#include "MobileNetSSD.h"
#include "MaskRCNN.h"


int main()
{
	runYOLOv3(0, "../models/yolov3.cfg", "../models/yolov3.weights", 1280, 720, 0.5, 0.4);
	runMobileNetSSD(0, "../models/ssd_mobilenet_v1_coco_11_06_2017.pb", "../models/ssd_mobilenet_v1_coco_11_06_2017.pbtxt", 1280, 720, 0.5);
	runMaskRCNN("test.jpg", "../models/mask_rcnn_inception_v2_coco_2018_01_28.pb", "../models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt", 0.5);
	return 0;
}