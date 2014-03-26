#include "testApp.h"

using namespace ofxCv;
using namespace cv;



vector<string> getEmotionTxts() {

    std::vector<string> txts;

    string kanadePath = "/dev/resource/kanade/";
    ofDirectory dirEmotions(kanadePath + "/Emotion");
    dirEmotions.listDir();

    for(int i = 0; i < dirEmotions.numFiles(); i++) {

        ofDirectory dirSubject(dirEmotions.getPath(i));
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

int getEmotion(string txt) {

    ofBuffer buffer = ofBufferFromFile(txt);
    vector<string> words = ofSplitString(buffer.getText(), " ", true, true);
    return (int)atof(words[0].c_str());

}

string getLastPic(string path) {
    ofStringReplace(path, "Emotion", "images");
    ofStringReplace(path, "_emotion.txt", ".png");
    return path;
}

vector<string> getTxtsWithEmotion(vector<string> txts, int emotionCode) {

    vector<string> result;

    for (int i = 0; i < txts.size(); i++)
        if (getEmotion(txts[i]) == emotionCode)
            result.push_back(txts[i]);

    return result;

}




void testApp::setup() {

    txts = getEmotionTxts();
    consideredEmotions = {1, 2, 3, 4, 5, 6, 7};

	ofSetVerticalSync(true);
	//cam.initGrabber(640, 480);

	tracker.setup();
	tracker.setRescale(.5);

    classifier.load("facs");

}

void testApp::update() {

    if (emotionId > consideredEmotions.size()) {
        //classifier.save("expressions");
        std::exit(0);
        return;
    }

    if (imageId >= txtsEmotion.size()) {
        cout << "Ratio: " << ratio/txtsEmotion.size() << "\n";
        ratio = 0;
        cout << "Current Emotion: " << consideredEmotions[emotionId] << "\n";
        txtsEmotion = getTxtsWithEmotion(txts, consideredEmotions[emotionId++]);
        if (txtsEmotion.size() == 0) {
            imageId = 999;
            return;
        }
        imageId = 0;
    }

    if (frame == 0) {
        string filename = getLastPic(txtsEmotion[imageId++]);
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
        ratio += (maxEmotion == emotionId) ? 1 : 0;
        cout << "\t(" << maxEmotion << ", " << emotionId << "), " << "(" << probs[maxEmotion] << ", " << probs[emotionId] << ")\n";
        frame = 0;
    }

}

void testApp::draw() {

	ofSetColor(255);
	//cam.draw(0, 0);

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

        probs[i] = 0;
        float weights = 0;
        for (int j = 0; j < 20; j++) {
            weights += emotions[i][j];
            probs[i] += classifier.getProbability(j) * emotions[i][j];
        }
        probs[i] /= weights;

        ofSetColor(ofColor::red);
        ofRect(ofGetWidth() - 120, 0, w * probs[i] * 4 + 0.5, h);
        ofSetColor(255);
        ofDrawBitmapString(emotionNames[i], ofGetWidth() - 120, h);
        ofTranslate(0, h + 5);

        float maxProb = 0;
        maxEmotion = 0;
        for (int i = 0; i < 7; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxEmotion = i + 1;
            }
        }

    }

	ofPopMatrix();
	ofPopStyle();

}

void testApp::keyPressed(int key) {}
