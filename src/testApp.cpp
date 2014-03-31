#include "testApp.h"

using namespace ofxCv;
using namespace cv;

//--------------------------------------------------------------
void testApp::setup() {

	int x = 640, y = 0;

	ofSetVerticalSync(true);
	ofEnableSmoothing();

	// facetracking initialization
	cam.initGrabber(640, 480);
	tracker.setup();
	tracker.setRescale(.5);
    classifier.load("emotions");

    // set some sketch parameters
    meshColor = ofColor(32, 225, 205);
    for (int i = 0; i < 7; i++)
    	probs[i] = 0;
    string emotionLabels[7] = {"Angry", "Contempt", "Disgust", "Fear", "Happy", "Sadness", "Surprise"};

    // initialize GUIs
    gui1 = new ofxUISuperCanvas("MESH");
    gui1->addSpacer();
    gui1->addLabel("Visibility", OFX_UI_FONT_MEDIUM);
    gui1->addSpacer();
	gui1->addToggle("Show", meshView[0]);
	gui1->addToggle("Complex", meshView[1]);
    gui1->addSpacer();
    gui1->addLabel("Color", OFX_UI_FONT_MEDIUM);
    gui1->addSpacer();
    gui1->addSlider("Red", 0, 255, meshColor.r);
    gui1->addSlider("Green", 0, 255, meshColor.g);
    gui1->addSlider("Blue", 0, 255, meshColor.b);
    gui1->setPosition(x, y);
    gui1->autoSizeToFitWidgets();
    ofAddListener(gui1->newGUIEvent,this,&testApp::guiEvent);

    y += 122;
    gui2 = new ofxUISuperCanvas("EMOTIONS");
    gui2->addSpacer();
    for (int i = 0; i < 7; i++)
    	gui2->addSlider(emotionLabels[i], 0, 1, &probs[i]);
    gui2->setPosition(640-212, 0);
    gui2->autoSizeToFitWidgets();

    /*
    gui3 = new ofxUISuperCanvas("INFO");
    gui3->addSpacer();
    gui3->addFPS();
    gui3->setPosition(0, 0);
    gui3->autoSizeToFitWidgets();
	*/

    y += 122;
    gui4 = new ofxUISuperCanvas("TRACKING");
    gui4->addSpacer();
	gui4->add2DPad("Position", ofPoint(0, 640), ofPoint(0, 480), &positionPoint);
    gui4->addSpacer();
    gui4->addCircleSlider("Scale", 0, 10, &scale);
    gui4->setPosition(0, 0);
    gui4->autoSizeToFitWidgets();

}

//--------------------------------------------------------------
void testApp::update() {

	cam.update();
	if(cam.isFrameNew()) {
		if(tracker.update(toCv(cam))) {

			ofVec2f position = tracker.getPosition();
			positionPoint = ofPoint(position.x, position.y);
			cout << position.x << ", " << position.y << endl;
			scale = tracker.getScale();

			classifier.classify(tracker);

		}
	}

}

//--------------------------------------------------------------
void testApp::draw(){

	ofSetColor(255);
	ofDrawBitmapString(ofToString((int) ofGetFrameRate()), 10, 20);
	cam.draw(0, 0);

	ofSetColor(meshColor);
	if (meshView[0]) {
		if (meshView[1])
			tracker.getImageMesh().drawWireframe();
		else
			tracker.draw();
	}

	int n = classifier.size();
	int primary = classifier.getPrimaryExpression();
    for (int i = 0; i < n; i++) {
    	probs[i] = classifier.getProbability(i);
    }

}

//--------------------------------------------------------------
void testApp::keyPressed(int key){

}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){

}

//--------------------------------------------------------------
void testApp::guiEvent(ofxUIEventArgs &e) {

	string n = e.getName();
	int kind = e.getKind();
	cout << "got event from: " << n << ", " << kind << endl;

	if (kind == 4) { // Mesh color

		ofxUISlider *rslider = (ofxUISlider *) e.widget;
		int value = rslider->getScaledValue();

		if (n == "Red")
			meshColor.r = value;
		else if (n == "Green")
			meshColor.g = value;
		else if (n == "Blue")
			meshColor.b = value;

	}

	else if (kind == 2) { // Mesh visibility

		ofxUIToggle *toggle = (ofxUIToggle *) e.getToggle();
		bool value = toggle->getValue();

		if (n == "Show")
			meshView[0] = value;
		else if (n == "Complex")
			meshView[1] = value;

	}

}
