#include "testApp.h"

using namespace ofxCv;
using namespace cv;

//--------------------------------------------------------------
void testApp::setup() {

	ofSetVerticalSync(true);
	ofEnableSmoothing();

    linear = (arguments.size() >= 3 ? ((int)atof(arguments[2].c_str()) == 1 ? true : false) : true);
    databasePath = (arguments.size() >= 4 ? arguments[3] : "kanadeSelectedPlusKdefAll");

	// facetracking initialization
	tracker.setup();
	tracker.setRescale(.5);
    classifier.load(databasePath);

    if (arguments.size() >= 2) {
        useImage = true;
        image.loadImage(arguments[1]);
    }
    else
        cam.initGrabber(640, 470);

    // Start preparing training input
    int TOTAL_SAMPLES = 0;
    int index = 0;

    for (int i = 0; i < EMOTION_COUNT; i++)
        TOTAL_SAMPLES += classifier.getExpression(i).size();

    float labels[TOTAL_SAMPLES];
    float trainingData[TOTAL_SAMPLES][198];

    for (int i = 0; i < EMOTION_COUNT; i++) {
        Expression expression = classifier.getExpression(i);
        int sampleCount = expression.size();
        for (int j = 0; j < sampleCount; j++) {
            labels[index] = i;
            Mat& sampleMat = expression.getExample(j);
            vector<float> sampleVector;
            sampleMat.convertTo(sampleVector, CV_32FC1);
            for (int k = 0; k < 198; k++)
                trainingData[index][k] = sampleVector[k];
            index++;
        }
    }

    Mat labelsMat(TOTAL_SAMPLES, 1, CV_32FC1, labels);
    Mat trainingDataMat(TOTAL_SAMPLES, 198, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.C           = 0.1;
    params.svm_type    = CvSVM::C_SVC;
    params.gamma       = 1;
    params.degree      = 2;
    params.coef0       = 0;
    params.kernel_type = (linear == true) ? CvSVM::LINEAR : CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-1);

    // Train the SVM
    SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), params, 10);
    params = SVM.get_params();

    // set some sketch parameters
    meshColor = ofColor(32, 225, 205);
    for (int i = 0; i < EMOTION_COUNT; i++) {
    	probs[i] = 0;
        emotionFrames[i] = 0;
        emotionPercentages[i] = 0;
    }

    int camWidth = 640, panelWidth = 212;

    // initialize GUIs
    gui1 = new ofxUISuperCanvas("MESH");
    gui1->addSpacer();
    gui1->addLabel("Visibility", OFX_UI_FONT_MEDIUM);
    gui1->addSpacer();
	gui1->addToggle("Show", meshView[0]);
	gui1->addToggle("Complex", meshView[1]);
	//gui1->addToggle("Axis", meshView[2]);/
	//gui1->addToggle("Miniature", meshView[3]);
    gui1->addSpacer();
    gui1->addLabel("Color", OFX_UI_FONT_MEDIUM);
    gui1->addSpacer();
    gui1->addSlider("Red", 0, 255, meshColor.r);
    gui1->addSlider("Green", 0, 255, meshColor.g);
    gui1->addSlider("Blue", 0, 255, meshColor.b);
    gui1->setPosition(0, 0);
    gui1->autoSizeToFitWidgets();
    gui1->disable();
    ofAddListener(gui1->newGUIEvent,this,&testApp::guiEvent);

    gui2 = new ofxUISuperCanvas("EMOTIONS");
    gui2->addSpacer();
    for (int i = 0; i < EMOTION_COUNT; i++)
    	gui2->addSlider(emotionLabels[i], 0, 1, &probs[i]);
    gui2->setPosition(camWidth - panelWidth, 0);
    gui2->autoSizeToFitWidgets();
    gui2->disable();

    gui3 = new ofxUISuperCanvas("MAIN");
    gui3->addSpacer();
    gui3->addTextInput("Emotion", "Face not tracked");
    gui3->addSpacer();
    gui3->addSlider("Variance", 0, 0.15, &stdDeviation);
    gui3->setPosition(camWidth/3, 0);
    gui3->autoSizeToFitWidgets();

    gui4 = new ofxUISuperCanvas("TRACKING");
    gui4->addSpacer();
	gui4->add2DPad("Position", ofPoint(0, 640), ofPoint(0, 470), &positionPoint);
    gui4->addSpacer();
    gui4->addCircleSlider("Scale", 0, 10, &scale);
    gui4->setPosition(camWidth, 0);
    gui4->autoSizeToFitWidgets();
    gui4->disable();

    gui5 = new ofxUICanvas();
    gui5->addLabel("Press i to toggle advanced panels");
    gui5->setPosition(0, 470 - 20);
    gui5->autoSizeToFitWidgets();

    gui6 = new ofxUISuperCanvas("TIME OCCURRENCES");
    gui6->addSpacer();
    for (int i = 0; i < EMOTION_COUNT + 1; i++) {
        if (i == 1)
            gui6->setWidgetPosition(OFX_UI_WIDGET_POSITION_RIGHT);
    	gui6->addSlider(emotionLabelsSmall[i], 0, 1, &emotionPercentages[i], 24, 64);
    }
	gui6->setWidgetPosition(OFX_UI_WIDGET_POSITION_DOWN);
    gui6->setPosition(camWidth - panelWidth, 470 - 110);
    gui6->autoSizeToFitWidgets();
    gui6->disable();

    gui7 = new ofxUISuperCanvas("OPTIONS");
    gui7->addSpacer();
    gui7->addSlider("Neutral threshold", 0, 1, &neutralThreshold);
    gui7->setPosition(0, 270);
    gui7->autoSizeToFitWidgets();
    gui7->disable();

}

//--------------------------------------------------------------
void testApp::update() {

	(useImage) ? image.update() : cam.update();

    if ((!useImage && tracker.update(toCv(cam))) || (useImage && tracker.update(toCv(image)))) {
        ofVec2f position = tracker.getPosition();
        positionPoint = ofPoint(position.x, position.y);
        scale = tracker.getScale();
        orientation = tracker.getOrientation();
        rotationMatrix = tracker.getRotationMatrix();
        classifier.classify(tracker);
    }

}

//--------------------------------------------------------------
void testApp::draw(){

	ofBackground(ofColor(170, 200, 200));
	ofSetColor(255);
	ofDrawBitmapString(ofToString((int) ofGetFrameRate()), 10, 20);

    (useImage) ? image.draw(0, 0) : cam.draw(0, 0);

	Mat pointsMat;
	tracker.getObjectPointsMat().convertTo(pointsMat, CV_32FC1);
	norm(pointsMat);

	if (pointsMat.rows == 198)
        primaryExpression = SVM.predict(pointsMat);

	ofSetColor(meshColor);
	if (meshView[0]) {
		if (meshView[1])
			tracker.getImageMesh().drawWireframe();
		else
			tracker.draw();
	}

    int primary = classifier.getPrimaryExpression();

    int avgProb = 0;
    bool allAreReallySmall = true, allAreSmall = true;
    for (int i = 0; i < EMOTION_COUNT; i++) {
        probs[i] = classifier.getProbability(i);
        if (probs[i] > neutralThreshold)
            allAreSmall = false;
            primaryExpression = primary;
        avgProb += probs[i];
    }
    avgProb /= EMOTION_COUNT;

    stdDeviation = 0;
    for (int i = 0; i < EMOTION_COUNT; i++) {
        stdDeviation += pow(avgProb - probs[i], 2);
    }
    stdDeviation /= EMOTION_COUNT;

    if (allAreSmall)
        primaryExpression = 6;

    emotionFrames[primaryExpression]++;
    int totalCounts = 0;
    for (int i = 0; i < EMOTION_COUNT + 1; i++)
        totalCounts += emotionFrames[i];
    for (int i = 0; i < EMOTION_COUNT + 1; i++)
        emotionPercentages[i] = emotionFrames[i]/totalCounts;

    string primaryLabel = emotionLabels[primaryExpression];
    ((ofxUITextInput *) gui3->getWidget("Emotion"))->setTextString(primaryLabel);

    if (meshView[3] && !useImage) {
        ofSetColor(255);
        ofMesh objectMesh = tracker.getObjectMesh();
        ofMesh meanMesh = tracker.getMeanObjectMesh();
        ofSetupScreenOrtho(640, 480, OF_ORIENTATION_DEFAULT, true, -1000, 1000);
        cam.getTextureReference().bind();
        meanMesh.draw();
        cam.getTextureReference().unbind();
    }

}

//--------------------------------------------------------------
void testApp::keyPressed(int key){

    switch (key) {
        case 'i':
            gui1->toggleVisible();
            gui2->toggleVisible();
            gui4->toggleVisible();
            gui6->toggleVisible();
            gui7->toggleVisible();
            break;
        case 'k':
            cout << "age: " << tracker.getAge() << endl;
            cout << "found: " << tracker.getFound() << endl;
            cout << "orientation: " << tracker.getOrientation() << endl;
            cout << "direction: " << tracker.getDirection() << endl;
            break;
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

//--------------------------------------------------------------
void testApp::guiEvent(ofxUIEventArgs &e) {

	string n = e.getName();
	int kind = e.getKind();
	//cout << "got event from: " << n << ", " << kind << endl;

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
		else if (n == "Axis")
			meshView[2] = value;

	}

}
