#include "testApp.h"

using namespace ofxCv;
using namespace cv;

void testApp::setup() {

    //Emotion happy("happy", {0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0});

	ofSetVerticalSync(true);
	cam.initGrabber(640, 480);

	tracker.setup();
	tracker.setRescale(.5);

}

void testApp::update() {
	cam.update();
	if(cam.isFrameNew()) {
		if(tracker.update(toCv(cam))) {
			classifier.classify(tracker);
		}
	}
}

void testApp::draw() {

	ofSetColor(255);
	cam.draw(0, 0);

	//tracker.draw();
	ofSetColor(10,232,232);
	tracker.getImageMesh().drawWireframe();

	int w = 100, h = 12;
	ofPushStyle();
	ofPushMatrix();
	ofTranslate(5, 10);
	int n = classifier.size();
	int primary = classifier.getPrimaryExpression();
    for(int i = 0; i < n; i++){
        facProbabilities[i] = classifier.getProbability(i);
        ofSetColor(i == primary ? ofColor::red : ofColor::black);
        ofRect(0, 0, w * facProbabilities[i] + .5, h);
        ofSetColor(255);
        ofDrawBitmapString(classifier.getDescription(i), 5, 9);
        ofTranslate(0, h + 5);
    }
	ofPopMatrix();
	ofPopStyle();

	ofPushStyle();
	ofPushMatrix();
	ofTranslate(5, 10);


    vector<double> emProb;
    emProb.push_back((classifier.getProbability(4) + classifier.getProbability(8))/2);
    emProb.push_back((classifier.getProbability(2) + classifier.getProbability(3) + classifier.getProbability(5) + classifier.getProbability(14))/4);

    for (int i = 0; i < emProb.size(); i++) {
        ofSetColor(ofColor::red);
        ofRect(ofGetWidth() - 120, 0, w * emProb[i] + 0.5, h);
        ofSetColor(255);
        ofDrawBitmapString("blah", ofGetWidth() - 120, h);
        ofTranslate(0, h + 5);
    }

	ofPopMatrix();
	ofPopStyle();


	ofDrawBitmapString(ofToString((int) ofGetFrameRate()), ofGetWidth() - 20, ofGetHeight() - 10);
	drawHighlightString(
		string() +
		"r - reset\n" +
		"e - add expression\n" +
		"a - add sample\n" +
		"s - save expressions\n"
		"l - load expressions",
		14, ofGetHeight() - 7 * 12);
}

void testApp::keyPressed(int key) {
	if(key == 'r') {
		tracker.reset();
		classifier.reset();
	}
	if(key == 'e') {
		classifier.addExpression();
	}
	if(key == 'a') {
		classifier.addSample(tracker);
	}
	if(key == 's') {
		classifier.save("expressions");
	}
	if(key == 'l') {
		classifier.load("expressions");
	}
	if(key == 'k') {
		emotionClassifier.load("emotions");
		emotionClassifier.addEmotion();
		emotionClassifier.addSample(facProbabilities);
		emotionsClassifier.save("emotions");
	}
}
