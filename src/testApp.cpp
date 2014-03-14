#include "testApp.h"
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atoi */
#include <cstring>
#include <string>

using namespace ofxCv;
using namespace cv;


vector<string> getFacsTxts() {

    std::vector<string> txts;

    string kanadePath = "/dev/resource/kanade/";
    ofDirectory dirFacs(kanadePath + "/FACS");
    dirFacs.listDir();

    for(int i = 0; i < dirFacs.numFiles(); i++) {

        ofDirectory dirSubject(dirFacs.getPath(i));
        dirSubject.listDir();
         for(int j = 0; j < dirSubject.numFiles(); j++) {
            ofDirectory dir(dirSubject.getPath(j));
            dir.listDir();
            for(int k = 0; k < dir.numFiles(); k++) {
                txts.push_back(dir.getPath(k));
                //ofLogNotice(dir.getPath(k));
            }
        }

    }

    return txts;

}

vector<int> getFacs(string txt) {

    vector<int> facs;
    ofBuffer buffer = ofBufferFromFile(txt);
    vector<string> words = ofSplitString(buffer.getText(), " ", true, true);

    for (int i = 0; i < words.size(); i += 2) {
        facs.push_back((int)atof(words[i].c_str()));
    }

    return facs;

}

string getLastPic(string path) {
    ofStringReplace(path, "FACS", "images");
    ofStringReplace(path, "_facs.txt", ".png");
    return path;
}

vector<string> getTxtsWithFac(vector<string> txts, int facCode) {

    vector<string> result;

    for (int i; i < txts.size(); i++) {

        vector<int> facs = getFacs(txts[i]);

        for (int j = 0; j < facs.size(); j++)
            if (facs[j] == facCode)
                result.push_back(txts[i]);

    }

    return result;

}


void testApp::setup() {

    txts = getFacsTxts();
    consideredFacs = {1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 27, 43};
    //consideredFacs = {1};

	ofSetVerticalSync(true);

	tracker.setup();
	tracker.setRescale(.5);

}

void testApp::update() {

    if (facId > consideredFacs.size()) {
        classifier.save("expressions");
        std::exit(0);
        return;
    }

    if (imageId >= txtsFac.size()) {
        cout << "Current FAC: " << consideredFacs[facId] << "\n";
        txtsFac = getTxtsWithFac(txts, consideredFacs[facId++]);
        imageId = 0;
        classifier.addExpression();
    }

    if (frame == 0) {
        string filename = getLastPic(txtsFac[imageId++]);
        ifstream ifile(filename);
        if (ifile)
            image.loadImage(filename);
    }

    image.update();
    tracker.update(toCv(image));

    if (frame < 2) {
        position = tracker.getPosition();
        scale = tracker.getScale();
        orientation = tracker.getOrientation();
        rotationMatrix = tracker.getRotationMatrix();
        classifier.classify(tracker);
        frame++;
    }
    else if (frame == 2) {
        classifier.addSample(tracker);
        frame = 0;
    }

}

void testApp::draw() {

	ofSetColor(255);

	image.draw(0, 0);

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
            ofSetColor(i == primary ? ofColor::red : ofColor::black);
            ofRect(0, 0, w * classifier.getProbability(i) + .5, h);
            ofSetColor(255);
            ofDrawBitmapString(classifier.getDescription(i), 5, 9);
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
}
