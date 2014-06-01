#include "testApp.h"

using namespace ofxCv;
using namespace cv;


//--------------------------------------------------------------
// Function that returns a vector with the path of all files
// inside the KDEF folder
// Please make sure to delete all files that don't end with s.jpg
vector<string> getImagePaths() {

    std::vector<string> paths;

    string kdefPath = "/dev/resource/kdef/";
    ofDirectory dir(kdefPath);
    dir.listDir();

    for (int i = 0; i < dir.numFiles(); i++) {
        ofDirectory dirSubject(dir.getPath(i));
        dirSubject.listDir();
        for (int j = 0; j < dirSubject.numFiles(); j++)
            paths.push_back(dirSubject.getPath(j));
    }

    return paths;

}


vector<string> filterByEmotion(vector<string> paths, string code) {

    vector<string> selected;

	for (int i = 0; i < paths.size(); i++) {
		string imageCode = paths[i].substr(paths[i].length() - 7, 2);
		if (imageCode.compare(code) == 0)
            selected.push_back(paths[i]);
	}

    return selected;

}



//--------------------------------------------------------------
void testApp::setup() {

	ofSetVerticalSync(true);
	ofEnableSmoothing();

	// Get the path of all images
	paths = getImagePaths();

	// Facetracking initialization
	tracker.setup();
	tracker.setRescale(.5);

	// Initialize classifier
    classifier.load(databasePath);

}

//--------------------------------------------------------------
void testApp::update() {

	if (sPaths.size() == 0 || currentImage >= sPaths.size()) {
		sPaths = filterByEmotion(paths, emotionCodes[++currentEmotion]);
		currentImage = 0;
		if (currentEmotion <= EMOTION_COUNT - 1) {
            classifier.addExpression();
            cout << "\nCurrent emotion: " << currentEmotion << ", " << emotionCodes[currentEmotion] << endl;
		}
        else {
            classifier.save(databasePath);
            std::exit(0);
        }
	}

    if (frame == 0) {
        string filename = sPaths[currentImage++];
        ifstream ifile(filename);
        if (ifile)
            image.loadImage(filename);
    }

	image.update();
	tracker.update(toCv(image));

    if (frame == 3) {
        classifier.addSample(tracker);
    	frame = -1;
    }

	frame++;

}

//--------------------------------------------------------------
void testApp::draw(){

	ofBackground(ofColor(170, 200, 200));
	ofSetColor(255);
	ofDrawBitmapString(ofToString((int) ofGetFrameRate()), 10, 20);

    image.draw(0, 0);

	ofSetColor(ofColor(32, 225, 205));
	tracker.getImageMesh().drawWireframe();

}

//--------------------------------------------------------------
void testApp::keyPressed(int key){

    switch (key) {
        default:
            break;
    }

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
