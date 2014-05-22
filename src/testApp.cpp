#include "testApp.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace ofxCv;
using namespace cv;

//--------------------------------------------------------------
void testApp::setup(){

    // Data for visual representation
    int width = 512, height = 512;
    image = Mat::zeros(height, width, CV_8UC3);

    classifier.load("emotions");

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
            sampleMat.row(0).copyTo(sampleVector);
            for (int k = 0; k < 198; k++)
                trainingData[index][k] = sampleVector[k];
            index++;
        }
    }

    cout << TOTAL_SAMPLES << endl;

    Mat labelsMat(TOTAL_SAMPLES, 1, CV_32FC1, labels);
    Mat trainingDataMat(TOTAL_SAMPLES, 198, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.C           = 0.1;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

}

//--------------------------------------------------------------
void testApp::update(){


}

//--------------------------------------------------------------
void testApp::draw(){

    //ofxCv::drawMat(image, 0, 0, 640, 480);

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
