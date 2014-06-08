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
		ofVec3f orientation;
		ofMatrix4x4 rotationMatrix;

		float probs[7];
		float emotionFrames[7];
		float emotionPercentages[7];
		string emotionLabels[8] = {"Angry", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"};
		string emotionLabelsSmall[8] = {"An", "Di", "Fe", "Ha", "Sa", "Su", "Ne"};

		// UI related
		ofxUISuperCanvas *gui1;
		ofxUISuperCanvas *gui2;
		ofxUISuperCanvas *gui3;
		ofxUISuperCanvas *gui4;
		ofxUICanvas *gui5;
		ofxUISuperCanvas *gui6;
		ofxUISuperCanvas *gui7;
		void guiEvent(ofxUIEventArgs &e);

		ofColor meshColor; // RGB
		bool meshView[4] = {true, true, false, false}; // Show, complex, axis, miniature
		ofPoint positionPoint = ofPoint(0,0);
		float scale = 0;
		float stdDeviation = 0;
		float neutralThreshold = 0.45;

		// Image
		ofImage image;
        bool useImage = false;

        // SVM
    	CvSVM SVM;
    	int primaryExpression = 0;
    	bool linear;

    	string databasePath;
    	int const EMOTION_COUNT = 6;
        vector<string> arguments;


};
