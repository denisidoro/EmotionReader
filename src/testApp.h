#pragma once

#include "ofMain.h"
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

	double facProbabilities[21];
	double emotions[7][20] = {
        {0.0692233, 0.0394132, 0.105007, 0.321104, 0.0200476, 0.0360923, 1.18297, 0.0571372, 0.165437, 0.0191974, 0.0345871, 0.00321583, 0.233588, 0.0525605, 0.0492818,0.379198, 0.0717249, 0.00834457, 0.0718313, 0.0355944},
        {0.00471227, 5.56952e-014, 0.0440207, 0.0127622, 2.98946e-011, 5.01601e-007, 2.19314, 0.0017416, 1.99608e-005, 3.47707e-011, 1.0923e-013, 5.34471e-010, 5.16983e-005, 2.12662e-011, 5.85422e-005, 0.610826, 6.89492e-005, 0.00232038, 0.0191696,1.70147e-008},
        {0.0568409, 0.0123355, 0.0600161, 0.34212, 0.00615451, 0.0592867, 0.103146, 0.107549, 0.211274, 0.00736939, 0.0113454, 0.00510233, 0.513387, 0.0208716, 0.33996,0.34976, 0.468897, 0.0494446, 0.152774, 0.0884685},
        {0.444567, 0.0516496, 0.0658078, 0.0832292, 0.000782017, 0.18876, 0.0155735, 0.0182244, 0.771609, 0.0499698, 0.00361979, 0.00410986, 0.557812, 0.0460062, 0.124492, 0.104306, 0.0324187, 0.0169377, 0.175055, 0.16507},
        {0.071005, 0.0109183, 0.0204952, 0.0631153, 0.000149872, 0.154055, 0.0176113, 0.0183456, 0.639217, 0.0604137, 0.000141066, 0.00258459, 0.163226, 0.0168234, 0.745374, 0.184994, 0.0597511, 0.0306928, 0.66286, 0.049242},
        {0.280289, 0.115001, 0.278184, 0.284719, 0.0201938, 0.138013, 0.0378411, 0.0562755, 0.535455, 0.122869, 0.0215161, 0.0127346, 0.285022, 0.0442786, 0.0516211, 0.0946621, 0.0410869, 0.0174083, 0.370546, 0.120854},
        {0.400294, 0.556488, 0.0430079, 0.0355124, 0.00498915, 0.0197266, 0.00437082, 0.00483763, 0.381133, 0.103575, 0.494007, 0.00853597, 0.0301029, 0.839436, 0.0192729, 0.00535973, 0.000716416, 0.00132467, 0.0189431, 0.00427111}
	};
	string emotionNames[7] = {"anger", "fear", "disgust", "contempt", "joy", "sadness", "surprise"};

};
