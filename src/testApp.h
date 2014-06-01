#pragma once

#include "ofMain.h"
#include "ofxUI.h"
#include "ofxFaceTracker.h"

class testApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		// OpenCV-related
		ofVideoGrabber cam;
		ofxFaceTracker tracker;
		ExpressionClassifier classifier;

		string emotionCodes[8] = {"AN", "DI", "AF", "HA", "SA", "SU", "NE"};
		const int EMOTION_COUNT = 6;
		string databasePath = "kdefAll";

		// Image
		ofImage image;

    	// Test-related
    	vector<string> paths;
    	vector<string> sPaths;
    	int currentEmotion = -1;
    	int currentImage;
    	int frame = 0;

        vector<string> arguments;

};
