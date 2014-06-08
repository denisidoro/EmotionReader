#include "testApp.h"

using namespace ofxCv;
using namespace cv;


int kanadeGetEmotion(string txt) {
    ofBuffer buffer = ofBufferFromFile(txt);
    vector<string> words = ofSplitString(buffer.getText(), " ", true, true);
    return (int)atof(words[0].c_str()) - 1;
}

string kanadeGetLastPic(string path) {
    ofStringReplace(path, "Emotion", "images");
    ofStringReplace(path, "_emotion.txt", ".png");
    return path;
}



//--------------------------------------------------------------
// Function that returns a vector with the path of all files
// If using KDEF, please make sure to delete all files that don't end with s.jpg
vector<string> testApp::getImagePaths(int db) {

    std::vector<string> paths;

    if (db == 0) {

        string kdefPath = "/dev/resource/kdef/";
        ofDirectory dir(kdefPath);
        dir.listDir();

        for (int i = 0; i < dir.numFiles(); i++) {
            ofDirectory dirSubject(dir.getPath(i));
            dirSubject.listDir();
            for (int j = 0; j < dirSubject.numFiles(); j++)
                paths.push_back(dirSubject.getPath(j));
        }

    }

    else {

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
                    paths.push_back(dir.getPath(k));
                }
            }

        }

    }

    return paths;

}


vector<string> testApp::filterBySubject(vector<string> paths, int id) {

    vector<string> selected;

    for (int i = (EMOTION_COUNT + 1)*id; i < (EMOTION_COUNT + 1)*(id+1); i++) {
        if ((i - (EMOTION_COUNT + 1)*id + 1) % 5 != 0)
            selected.push_back(paths[i]);
    }

    return selected;

}


vector<string> testApp::filterByEmotion(vector<string> paths, string code, int db) {

    vector<string> selected;

    if (db == 0) {

        for (int i = 0; i < paths.size(); i++) {
            string imageCode = paths[i].substr(paths[i].length() - 7, 2);
            if (imageCode.compare(code) == 0)
                selected.push_back(paths[i]);
        }

    }

    else {

        int conversion[6] = {0, 2, 3, 4, 5, 6};
        for (int i = 0; i < paths.size(); i++) {
            if (kanadeGetEmotion(paths[i]) == conversion[currentEmotion])
                selected.push_back(kanadeGetLastPic(paths[i]));
        }

    }

    return selected;

}



//--------------------------------------------------------------
void testApp::setup() {

	ofSetVerticalSync(true);
	ofEnableSmoothing();

	// Set variables
    for (int i = 0; i < EMOTION_COUNT; i++)
        for (int j = 0; j < EMOTION_COUNT; j++)
            occurrences[i][j] = 0;

    testingDatabase = (arguments.size() >= 2 ? ::atof(arguments[1].c_str()) : 0);
    databasePath = (arguments.size() >= 3 ? arguments[2] : "kdefAll");

	// Get the path of all images
	paths = getImagePaths(testingDatabase);

	// Facetracking initialization
	tracker.setup();
	tracker.setRescale(.5);

    // Set up SVM's parameters
    params.C           = 0.1;
    params.svm_type    = CvSVM::C_SVC;
    params.gamma       = 1;
    params.degree      = 2;
    params.coef0       = 0;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-1);

}

//--------------------------------------------------------------
void testApp::update() {

	if (sPaths.size() == 0 || currentEmotion > sPaths.size()) {

		if (currentSubject < SUBJECT_COUNT - 1) {

            /* START TRAINING */

            currentSubject++;

            classifier.reset();
            classifier.load(databasePath, currentSubject);

            // Start preparing training input
            int TOTAL_SAMPLES = 0;
            int index = 0;
            int expressionCount = classifier.size();

            for (int i = 0; i < expressionCount; i++)
                TOTAL_SAMPLES += classifier.getExpression(i).size();

            float labels[TOTAL_SAMPLES];
            float trainingData[TOTAL_SAMPLES][198];

            for (int i = 0; i < expressionCount; i++) {
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

            // Train the SVM
            if (currentSubject == 0)
            SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), params, 10);

            /* FINISH TRAINING */

            sPaths.clear();
            sPaths = filterBySubject(paths, currentSubject);
            cout << endl;
            for (auto p : sPaths)
                cout << p << endl;

            frame = 0;
            currentEmotion = 0;

            cout << "\nCurrent subject: " << currentSubject << endl;

		}

        else {

		    ofDirectory dir("results/");
		    dir.listDir();
			stringstream ss;
			ss << dir.numFiles() - 2;

			ofFile file(ofToDataPath("results/" + ss.str() + ".txt"), ofFile::WriteOnly);
			file.create();

        	file << "testdatabase: " << testingDatabase << "\ntrainDatabase: " << databasePath << "\n\n";

            for (int i = 0; i < EMOTION_COUNT; i++) {

                int total = 0;
                for (int j = 0; j < EMOTION_COUNT; j++)
                total += occurrences[i][j];

                for (int j = 0; j < EMOTION_COUNT; j++) {
                    file << fixed << setprecision(1) << (float)occurrences[i][j]/total*100 << "\t";
                    if (j == EMOTION_COUNT - 1)
                        file << endl;
                }

            }

			file.close();
            std::exit(0);

        }
	}

	if (currentEmotion > 5 && frame == 0) {
        currentEmotion++;
        return;
	}

	//cout << "   frame: " << frame << ", currentEmotion: " << currentEmotion << endl;

    if (frame == 0) {
        bool success;
        string filename = sPaths[currentEmotion];
        currentEmotion++;
        if (!image.loadImage(filename))
            return;
    }
    else if (frame == 3) {

		Mat pointsMat;
		tracker.getObjectPointsMat().convertTo(pointsMat, CV_32FC1);
		norm(pointsMat);

		if (pointsMat.rows == 198) {
            int result = (int)SVM.predict(pointsMat, true);
            cout << "emotion: " << (currentEmotion - 1) << ", result: " << result << ", filename: " << sPaths[currentEmotion - 1] << endl;
	        occurrences[currentEmotion - 1][result]++;
		}

    	frame = -1;

    }

	image.update();
	tracker.update(toCv(image));
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
