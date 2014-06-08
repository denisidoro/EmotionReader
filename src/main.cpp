#include "ofMain.h"
#include "testApp.h"

//========================================================================
int main(int argc, char *argv[]){

    // setup the GL context
	ofSetupOpenGL(640, 470, OF_WINDOW);

	// create app
    testApp *app = new testApp();

    app->arguments = vector<string>(argv, argv + argc);

    // start the app
	ofRunApp(app);

}
