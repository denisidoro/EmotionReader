#pragma once

#include "ofMain.h"
#include "ofxFaceTracker.h"
//#include "Emotion.h"

class testApp : public ofBaseApp {
public:
	void setup();
	void update();
	void draw();
	void keyPressed(int key);

	ofVideoGrabber cam;
	ofxFaceTracker tracker;
	ExpressionClassifier classifier;

	double facProbabilities[21];

};
