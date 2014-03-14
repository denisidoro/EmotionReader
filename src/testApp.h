#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxFaceTracker.h"

class testApp : public ofBaseApp {
public:
	void setup();
	void update();
	void draw();
	void keyPressed(int key);

	ofVideoGrabber cam;
	ofxFaceTracker tracker;
	ExpressionClassifier classifier;

    ofVec2f position;
	float scale;
	ofVec3f orientation;
	ofMatrix4x4 rotationMatrix;
	//Mat translation, rotation;
	ofMatrix4x4 pose;

	ofImage image;

	int frame = 0;
	int imageId = 0;
	int emotionId = 0;
	vector<string> txts;
	vector<int> consideredEmotions;
	vector<string> txtsEmotion;

};
