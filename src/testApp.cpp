#include "testApp.h"

using namespace ofxCv;
using namespace cv;

void testApp::setup() {

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

    for (int i = 0; i < 7; i++) {

        float prob = 0, weights = 0;
        for (int j = 0; j < 20; j++) {
            weights += emotions[i][j];
            prob += classifier.getProbability(j) * emotions[i][j];
        }
        prob /= weights;

        ofSetColor(ofColor::red);
        ofRect(ofGetWidth() - 120, 0, w * prob + 0.5, h);
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
		classifier.save("facs");
	}
	if(key == 'l') {
		classifier.load("facs");
	}
}
