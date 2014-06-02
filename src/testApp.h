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

		vector<string> getImagePaths(int db);
		vector<string> filterByEmotion(vector<string> paths, string code, int db);
		vector<string> filterBySubject(vector<string> paths, int id);

		// OpenCV-related
		ofVideoGrabber cam;
		ofxFaceTracker tracker;
		ExpressionClassifier classifier;
		ofVec3f orientation;
		ofMatrix4x4 rotationMatrix;

		string emotionCodes[8] = {"AN", "DI", "AF", "HA", "SA", "SU", "NE"};
		const int EMOTION_COUNT = 6;
		string databasePath;

		// Image
		ofImage image;

        // SVM
    	CvSVM SVM;
        CvSVMParams params;

    	// Test-related
    	vector<string> paths;
    	vector<string> sPaths;
    	int occurrences[7][7];
    	int currentSubject = -1;
    	int currentEmotion;
    	int frame = 0;

        vector<string> arguments;
        int testingDatabase; // 0: KDEF, 1: Kanade
        const int SUBJECT_COUNT = 140;

};
