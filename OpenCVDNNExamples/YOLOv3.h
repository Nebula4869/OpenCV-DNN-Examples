#pragma once
void runYOLOv3(int cameraID, char* cfgFile, char* darknetModel, int frameWidth, int frameHeight, float scoreThreshold, float nmsThreshold);