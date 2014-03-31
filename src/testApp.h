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

		float probs[7];

		// UI related
		ofxUISuperCanvas *gui1;
		ofxUISuperCanvas *gui2;
		ofxUISuperCanvas *gui3;
		ofxUISuperCanvas *gui4;
		void guiEvent(ofxUIEventArgs &e);

		ofColor meshColor; // RGB
		bool meshView[2] = {true, true}; // Show, Complex
		ofPoint positionPoint = ofPoint(0,0);
		float scale = 0;

};
