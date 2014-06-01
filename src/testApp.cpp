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

	// Set variables
    for (int i = 0; i < EMOTION_COUNT; i++)
        for (int j = 0; j < EMOTION_COUNT; j++)
            occurrences[i][j] = 0;

    conf_c = (arguments.size() >= 2 ? ::atof(arguments[1].c_str()) : 0.1);
    conf_gamma = (arguments.size() >= 3 ? ::atof(arguments[2].c_str()) : 0.1);
    cout << "C: " << conf_c << ", gamma: " << conf_gamma << endl;

	// Get the path of all images
	paths = getImagePaths();

	// Facetracking initialization
	tracker.setup();
	tracker.setRescale(.5);
    classifier.load(databasePath);

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

    // Set up SVM's parameters
    CvSVMParams params;
    params.C           = conf_c;
    params.svm_type    = CvSVM::C_SVC;
    params.gamma       = conf_gamma;
    params.degree      = 2;
    params.coef0       = 0;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-1);

    // Train the SVM
    SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), params, 10);
    params = SVM.get_params();
    conf_c = params.C;
    conf_gamma = params.gamma;
    cout << "C: " << params.C << ", gamma: " << params.gamma << "\n\n" << params.degree;

}

//--------------------------------------------------------------
void testApp::update() {
	if (sPaths.size() == 0 || currentImage >= sPaths.size()) {
		sPaths = filterByEmotion(paths, emotionCodes[++currentEmotion]);
		currentImage = 0;
		if (currentEmotion <= EMOTION_COUNT - 1)
            cout << "\nCurrent emotion: " << currentEmotion << ", " << emotionCodes[currentEmotion] << endl;
        else {

		    ofDirectory dir("results/");
		    dir.listDir();
			stringstream ss;
			ss << dir.numFiles() - 2;

			ofFile file(ofToDataPath("results/" + ss.str() + ".txt"), ofFile::WriteOnly);
			file.create();
        	file << "database: " << databasePath << "\ngamma: " << conf_gamma << "\nc: " << conf_c << "\n\n";

            int total = 0;
            for (int i = 0; i < EMOTION_COUNT; i++)
                total += occurrences[0][i];

            for (int i = 0; i < EMOTION_COUNT; i++) {
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

    if (frame == 0) {
        string filename = sPaths[currentImage++];
        ifstream ifile(filename);
        if (ifile)
            image.loadImage(filename);
    }
    else if (frame == 3) {

		Mat pointsMat;
		tracker.getObjectPointsMat().convertTo(pointsMat, CV_32FC1);
		norm(pointsMat);

		if (pointsMat.rows == 198) {
            int result = (int)SVM.predict(pointsMat);
            cout << result << ", ";
	        occurrences[currentEmotion][result]++;
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
