#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include"glm.hpp"
#include"glimage.hpp"

#define  GL_GLEXT_PROTOTYPES

#if defined(__APPLE__) || defined(MACOSX)
# include <GLUT/glut.h> 
#else
# include <GL/glut.h>
#endif

#define xmax 10
#define ymax 10

#define xmlPath "haarcascade_frontalface_default.xml"

using namespace cv;
using namespace std;

GLfloat background[]={-xmax, -ymax, 0.0,
		1.0, 0.0, 0.0,
		1.0, 1.0,
		xmax, -ymax, 0.0,
		1.0, 0.0, 0.0,
		0.0, 1.0,
		xmax, ymax, 0.0,
		1.0, 0.0, 0.0,
		0.0, 0.0,
		-xmax, ymax, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0
};

GLfloat data[] = {/*
	-1.f, 1.f, 0.f,
	1.0, 0.0,
	1.f, 1.f, 0.f,
	0.0, 0.0,
	1.f,  -1.f, 0.f,
	0.0, 1.0,
	-1.f,  -1.f, 0.f,
	1.0, 1.0*/

	-1.f, -1.f, 0.f,
	1.0f, 1.0f,
	1.f, -1.f, 0.f,
	0.0f, 1.0f,
	1.f,  1.f, 0.f,
	0.0f, 0.0f,
	-1.f,  1.f, 0.f,
	1.0f, 0.0f
};

GLfloat 	swordVBO[32], bladeVBO[32];

static void initGL          (void);
static void reshape         (int, int);
static void drawBackground    (void);
static void display         (void);

static GLuint texture=0, hiltText = 0, bladeText = 0;

//CvCapture* capture;
VideoCapture *capture;

IplImage* frame;
Mat mat_frame;
CascadeClassifier* hand_cc;

Mat ci, gsi, imgOriginal;

GLMmodel *model=NULL;

int x=0, y=0;

void initCV(){
	//capture = cvCaptureFromCAM(CV_CAP_ANY);
		
	//if( !hand_cc.load(xmlPath) ){ printf("--(!)Error loading xml file\n");};
	
	hand_cc = new CascadeClassifier(xmlPath);
	
	capture = new VideoCapture(CV_CAP_ANY);
}

void initSwordPosition(double x, double y, double z, double width, double height){
	swordVBO[0] = x; swordVBO[1] = y; swordVBO[2] = z;
	swordVBO[3] = 0.0; swordVBO[4] = 1.0; swordVBO[5] = 0.0;
	swordVBO[6] = 0.0; swordVBO[7] = 1.0;

	swordVBO[8] = x+width; swordVBO[9] = y; swordVBO[10] = z;
	swordVBO[11] = 0.0; swordVBO[12] = 1.0; swordVBO[13] = 0.0;
	swordVBO[14] = 0.0; swordVBO[15] = 0.0;

	swordVBO[16] = x+width;	swordVBO[17] = y+height;	swordVBO[18] = z;
	swordVBO[19] = 0.0;	swordVBO[20] = 1.0;	swordVBO[21] = 0.0;
	swordVBO[22] = 1.0;	swordVBO[23] = 0.0;

	swordVBO[24] = x;	swordVBO[25] = y+height;	swordVBO[26] = z;
	swordVBO[27] = 0.0;	swordVBO[28] = 1.0;	swordVBO[29] = 0.0;
	swordVBO[30] = 1.0;	swordVBO[31] = 1.0;

	printf("swordVBO : (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) \n", swordVBO[0], swordVBO[1], swordVBO[2], swordVBO[6], swordVBO[7], swordVBO[8], swordVBO[12], swordVBO[13], swordVBO[14], swordVBO[18], swordVBO[19], swordVBO[20]);
	
	bladeVBO[0] = x; bladeVBO[1] = y+height; bladeVBO[2] = z;
	bladeVBO[3] = 0.0; bladeVBO[4] = 1.0; bladeVBO[5] = 0.0;
	bladeVBO[6] = 0.0; bladeVBO[7] = 1.0;

	bladeVBO[8] = x+width; bladeVBO[9] = y+height; bladeVBO[10] = z;
	bladeVBO[11] = 0.0; bladeVBO[12] = 1.0; bladeVBO[13] = 0.0;
	bladeVBO[14] = 0.0; bladeVBO[15] = 0.0;

	bladeVBO[16] = x+width;	bladeVBO[17] = y+3*height;	bladeVBO[18] = z;
	bladeVBO[19] = 0.0;	bladeVBO[20] = 1.0;	bladeVBO[21] = 0.0;
	bladeVBO[22] = 1.0;	bladeVBO[23] = 0.0;

	bladeVBO[24] = x;	bladeVBO[25] = y+3*height;	bladeVBO[26] = z;
	bladeVBO[27] = 0.0;	bladeVBO[28] = 1.0;	bladeVBO[29] = 0.0;
	bladeVBO[30] = 1.0;	bladeVBO[31] = 1.0;

	printf("bladeVBO : (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) \n", bladeVBO[0], bladeVBO[1], bladeVBO[2], bladeVBO[6], bladeVBO[7], bladeVBO[8], bladeVBO[12], bladeVBO[13], bladeVBO[14], bladeVBO[18], bladeVBO[19], bladeVBO[20]);
}

GLuint ConvertIplToTexture(IplImage *image)
{
  GLuint texture;

  glGenTextures(1,&texture);
  glBindTexture(GL_TEXTURE_2D,texture);
  /*glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_DECAL);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);*/
  gluBuild2DMipmaps(GL_TEXTURE_2D,3,image->width,image->height,
  GL_BGR,GL_UNSIGNED_BYTE,image->imageData);

 return texture;
}

// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint for use
GLuint matToTexture(cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter)
{
	// Generate a number for our textureID's unique handle
	GLuint textureID;
	glGenTextures(1, &textureID);

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Catch silly-mistake texture interpolation method for magnification
	if (magFilter == GL_LINEAR_MIPMAP_LINEAR  ||
	    magFilter == GL_LINEAR_MIPMAP_NEAREST ||
	    magFilter == GL_NEAREST_MIPMAP_LINEAR ||
	    magFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
		magFilter = GL_LINEAR;
	}

	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = GL_BGR;
	if (mat.channels() == 1)
	{
		inputColourFormat = GL_LUMINANCE;
	}

	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
	             0,                 // Pyramid level (for mip-mapping) - 0 is the top level
	             GL_RGB,            // Internal colour format to convert to
	             mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
	             mat.rows,          // Image height i.e. 480 for Kinect in standard mode
	             0,                 // Border width in pixels (can either be 1 or 0)
	             inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
	             GL_UNSIGNED_BYTE,  // Image data type
	             mat.ptr());        // The actual image data itself

	// If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
	if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
	    minFilter == GL_LINEAR_MIPMAP_NEAREST ||
	    minFilter == GL_NEAREST_MIPMAP_LINEAR ||
	    minFilter == GL_NEAREST_MIPMAP_NEAREST)
	{
		printf("glGenerateMipMap() \n");
		//glGenerateMipmap(GL_TEXTURE_2D);
	}

	return textureID;
}

void detectHand(){
	/*std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( mat_frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	  
	hand_cc.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	
	for (vector<Rect>::iterator fc = faces.begin(); fc != faces.end(); ++fc) {
		rectangle(mat_frame, (*fc).tl(), (*fc).br(), Scalar(0, 255, 0), 2, CV_AA);
	}
	  
	return mat_frame;*/

	vector<Rect> faces;
	*capture >> ci;
    cvtColor(ci, gsi, COLOR_BGR2GRAY);
    hand_cc->detectMultiScale(gsi, faces, 1.3, 5);
    for (vector<Rect>::iterator fc = faces.begin(); fc != faces.end(); ++fc) {
      rectangle(ci, (*fc).tl(), (*fc).br(), Scalar(0, 255, 0), 2, CV_AA);
    }
}

/*double convertCoord(int coord, int choice){
	double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	printf("xcam = %f ycam = %f\n", xmax_cam, ymax_cam);

	if(choice==0){
		printf("convert x = %f \n", (((2*coord*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1));
		return (((2*coord*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1);
	}else{
		printf("convert y = %f \n", (((2*(ymax_cam-coord*(ymax_cam/xmax_cam)))/ymax_cam)-1));
		return (((2*(ymax_cam-coord*(ymax_cam/xmax_cam)))/ymax_cam)-1);
	}
}*/

double convertCoord(int coord, int choice, double width, double height){
	double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	printf("xcam = %f ycam = %f\n", xmax_cam, ymax_cam);
	if(choice==0){
		printf("convert x = %f \n", ((((2*coord*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width);
		return ((((2*coord*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width;
	}else{
		if(choice==1)printf("convert y = %f \n", (((2*(ymax_cam-coord*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height);
		return (((2*(ymax_cam-coord*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height;
	}
}

double convertX(double x, double width, double height){
	double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	printf("convert x = %f \n", ((((2*x*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width);
	return ((((2*x*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width;
}

double convertY(double y, double width, double height){
	double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	printf("convert y = %f \n", (((2*(ymax_cam-y*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height);
	return (((2*(ymax_cam-y*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height;
}

double convertWidth(double toConvert){
	return ((2*toConvert)/imgOriginal.rows)+0.1;
}

double convertHeight(double toConvert){
	return ((2*toConvert)/imgOriginal.cols)+0.1;
}

/*void drawModel(double x, double y, double z){
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glDisable(GL_TEXTURE_2D);
	glColor3f(1.0, 1.0, 1.0);
	if(!model) {
		model = glmReadOBJ("obj/sword/Sword.obj");
		if (!model) exit(0);
			glmUnitize(model);
			glmScale(model, 15);

			glmFacetNormals(model);
			glmVertexNormals(model, 90.0);
	} 
	glmDraw(model, GLM_SMOOTH | GLM_MATERIAL);
	glPopAttrib();
}*/

void detectColor(){
	/*VideoCapture cap(0); //capture the video from web cam

	if(!cap.isOpened()){
		cout << "Cannot open the web cam" << endl;
		//return NULL;
	}*/

	int lowH = 10;
	int highH = 30;

	int lowS = 100; 
	int highS = 255;

	int lowV = 71;
	int highV = 255;

	Mat imgHSV, imgThresholded;

	*capture>>imgOriginal;

	/*if (!cap.read(imgOriginal)){
		cout << "Cannot read a frame from video stream" << endl;
	}*/

	cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

	inRange(imgHSV, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), imgThresholded); //Threshold the image

	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int thresh = 100;
	RNG rng(12345);
	int largest_area=0;
	int largest_contour_index=0;
	Rect bounding_rect;

	/// Detect edges using Threshold
	threshold( imgThresholded, threshold_output, thresh, 255, THRESH_BINARY );
	/// Find contours
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );

	for( int i = 0; i < contours.size(); i++ ){
		double a=contourArea( contours[i],false);  //  Find the area of contour
		if(a>largest_area){
			largest_area=a;
			largest_contour_index=i;                //Store the index of largest contour
			bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour

			printf(" x = %d \n", bounding_rect.x);
			printf(" y = %d \n", bounding_rect.y);
			printf(" width = %d \n", bounding_rect.width);
			printf(" height = %d \n", bounding_rect.height);
		}
	}

	Scalar color(255,255,255);
	rectangle(imgOriginal, bounding_rect, Scalar(0,255,0),1, 8,0);
	initSwordPosition(convertX(bounding_rect.x,convertWidth(bounding_rect.width),convertHeight(bounding_rect.height)),
							convertY(bounding_rect.y,convertWidth(bounding_rect.width),convertHeight(bounding_rect.height)), 
							0.1,
							convertWidth(bounding_rect.width),
							convertHeight(bounding_rect.height));
}

static void initGL(void){
	initCV();
	glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	glimageLoadAndBind("hilt.jpg", &hiltText);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glimageLoadAndBind("blade.jpg", &bladeText);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void reshape(int width, int height){
	glViewport(0,0,(GLsizei)(width),(GLsizei)(height));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (GLfloat)width/height, 0.01f, 5000.0f);	
	glMatrixMode(GL_MODELVIEW);
}

static void drawBackground(void){
	/*frame = cvQueryFrame(capture);
	texture = ConvertIplToTexture(frame);*/
	
	//mat_frame = cvQueryFrame(capture);
	//detectHand();
	detectColor();
	/*printf("mat.rows = %d \n",imgOriginal.rows);
	printf("mat.cols = %d \n",imgOriginal.cols);*/
	texture = matToTexture(imgOriginal, GL_NEAREST, GL_NEAREST, GL_CLAMP);

	glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	glBindTexture(GL_TEXTURE_2D, texture);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 8*sizeof(GLfloat), background);
	//glColorPointer(3, GL_FLOAT, 8*sizeof(GLfloat), &(background[3]));
	glTexCoordPointer(2, GL_FLOAT, 8*sizeof(GLfloat), &(background[6]));

	glDrawArrays(GL_QUADS, 0, 4);

	//glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

static void drawSword(){
	glBindTexture(GL_TEXTURE_2D, hiltText);
	glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 8*sizeof(GLfloat), swordVBO);
	//glColorPointer(3, GL_FLOAT, 6*sizeof(GLfloat), &(swordVBO[3]));
	glTexCoordPointer(2, GL_FLOAT, 8*sizeof(GLfloat), &(swordVBO[6]));
	
	glDrawArrays(GL_QUADS, 0, 4);

	//glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);	
}

static void drawBlade(){
	glBindTexture(GL_TEXTURE_2D, bladeText);
	glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 8*sizeof(GLfloat), bladeVBO);
	//glColorPointer(3, GL_FLOAT, 6*sizeof(GLfloat), &(bladeVBO[3]));
	glTexCoordPointer(2, GL_FLOAT, 8*sizeof(GLfloat), &(bladeVBO[6]));
	
	glDrawArrays(GL_QUADS, 0, 4);

	//glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

static void drawData(){
	detectColor();
	texture = matToTexture(imgOriginal, GL_NEAREST, GL_NEAREST, GL_CLAMP);
	glBindTexture(GL_TEXTURE_2D, texture);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 5*sizeof(GLfloat), data);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(GLfloat), &(data[3]));
	glDrawArrays(GL_QUADS, 0, 4);
	glDisableClientState(GL_VERTEX_ARRAY);	
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void display(void){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();					
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	drawData();
	//drawBackground();
	drawSword();
	drawBlade();

	/*glPushMatrix();
	glTranslatef(0.0, 0.0, 10.0);
	drawModel();
	glPopMatrix();*/

	glutSwapBuffers();
}

static void keyboard(unsigned char key, int x, int y){
	if (key == 27) exit(0);
}

int main(int argc, char **argv){
	glutInit(&argc, argv); 
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (500, 500); 
	glutInitWindowPosition (100, 100); 
	glutCreateWindow (argv[0]);
	initGL();  
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutIdleFunc(display);
	glutKeyboardFunc(keyboard);
	glutMainLoop(); 
	return 0; 
}
