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

#include"glimage.hpp"

#define  GL_GLEXT_PROTOTYPES
#define INFO if(0) printf

#if defined(__APPLE__) || defined(MACOSX)
# include <GLUT/glut.h> 
#else
# include <GL/glut.h>
#endif

using namespace cv;
using namespace std;

static void initGL          (void);
static void reshape         (int, int);
static void drawBackground    (void);
static void display         (void);

static GLuint camText=0, hiltText = 0, bladeText = 0;

VideoCapture *capture;

Mat imgOriginal;

GLfloat camVBO[] = {
	-1.f, -1.f, 0.f,
	1.0f, 1.0f,
	1.f, -1.f, 0.f,
	0.0f, 1.0f,
	1.f,  1.f, 0.f,
	0.0f, 0.0f,
	-1.f,  1.f, 0.f,
	1.0f, 0.0f
};

GLfloat 	swordVBO[20], bladeVBO[32];

void initCV(){
	capture = new VideoCapture(CV_CAP_ANY);
}

void initSwordPosition(double x, double y, double z, double width, double height){
	swordVBO[0] = x+0.2; swordVBO[1] = y; swordVBO[2] = z;
	swordVBO[3] = 0.0; swordVBO[4] = 1.0;

	swordVBO[5] = x+width+0.2; swordVBO[6] = y; swordVBO[7] = z;
	swordVBO[8] = 0.0; swordVBO[9] = 0.0;

	swordVBO[10] = x+width+0.2;	swordVBO[11] = y+height;	swordVBO[12] = z;
	swordVBO[13] = 1.0;	swordVBO[14] = 0.0;

	swordVBO[15] = x+0.2;	swordVBO[16] = y+height;	swordVBO[17] = z;
	swordVBO[18] = 1.0;	swordVBO[19] = 1.0;

	INFO("swordVBO : (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) \n", swordVBO[0], swordVBO[1], swordVBO[2], swordVBO[6], swordVBO[7], swordVBO[8], swordVBO[12], swordVBO[13], swordVBO[14], swordVBO[18], swordVBO[19]);
	
	bladeVBO[0] = x+0.2; bladeVBO[1] = y+height-0.1; bladeVBO[2] = z;
	bladeVBO[3] = 0.0; bladeVBO[4] = 0.0;

	bladeVBO[5] = x+width+0.2; bladeVBO[6] = y+height-0.1; bladeVBO[7] = z;
	bladeVBO[8] = 1.0; bladeVBO[9] = 0.0;

	bladeVBO[10] = x+width+0.2;	bladeVBO[11] = y+4*height;	bladeVBO[12] = z;
	bladeVBO[13] = 1.0;	bladeVBO[14] = 1.0;

	bladeVBO[15] = x+0.2;	bladeVBO[16] = y+4*height;	bladeVBO[17] = z;
	bladeVBO[18] = 0.0;	bladeVBO[19] = 1.0;

	INFO("bladeVBO : (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) \n", bladeVBO[0], bladeVBO[1], bladeVBO[2], bladeVBO[6], bladeVBO[7], bladeVBO[8], bladeVBO[12], bladeVBO[13], bladeVBO[14], bladeVBO[18], bladeVBO[19]);
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
		INFO("glGenerateMipMap() \n");
		//glGenerateMipmap(GL_TEXTURE_2D);
	}

	return textureID;
}

double convertX(double x, double width, double height){
	double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	INFO("convert x = %f \n", ((((2*x*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width);
	return ((((2*x*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width;
}

double convertY(double y, double width, double height){
	double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	INFO("convert y = %f \n", (((2*(ymax_cam-y*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height);
	return (((2*(ymax_cam-y*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height;
}

double convertWidth(double toConvert){
	return ((2*toConvert)/imgOriginal.rows)+0.1;
}

double convertHeight(double toConvert){
	return ((2*toConvert)/imgOriginal.cols)+0.1;
}

void detectColor(){
	int lowH = 10;
	int highH = 30;

	int lowS = 100; 
	int highS = 255;

	int lowV = 71;
	int highV = 255;

	Mat imgHSV, imgThresholded;

	*capture>>imgOriginal;

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

			INFO(" x = %d \n", bounding_rect.x);
			INFO(" y = %d \n", bounding_rect.y);
			INFO(" width = %d \n", bounding_rect.width);
			INFO(" height = %d \n", bounding_rect.height);
		}
	}

	Scalar color(255,255,255);
	//rectangle(imgOriginal, bounding_rect, Scalar(0,255,0),1, 8,0);
	initSwordPosition(convertX(bounding_rect.x,convertWidth(bounding_rect.width)+0.2,convertHeight(bounding_rect.height)+0.2),
							convertY(bounding_rect.y,convertWidth(bounding_rect.width)+0.2,convertHeight(bounding_rect.height)+0.2), 
							0.1,
							convertWidth(bounding_rect.width)+0.2,
							convertHeight(bounding_rect.height)+0.2);
}

static void initGL(void){
	initCV();
	glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	//for the transparency of the textures
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glimageLoadAndBind("hilt.png", &hiltText);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glimageLoadAndBind("blade.png", &bladeText);
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

static void drawHilt(){
	glBindTexture(GL_TEXTURE_2D, hiltText);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 5*sizeof(GLfloat), swordVBO);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(GLfloat), &(swordVBO[3]));
	
	glDrawArrays(GL_QUADS, 0, 4);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);	
}

static void drawBlade(){
	glBindTexture(GL_TEXTURE_2D, bladeText);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 5*sizeof(GLfloat), bladeVBO);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(GLfloat), &(bladeVBO[3]));
	
	glDrawArrays(GL_QUADS, 0, 4);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

static void drawCam(){
	detectColor();
	camText = matToTexture(imgOriginal, GL_NEAREST, GL_NEAREST, GL_CLAMP);
	glBindTexture(GL_TEXTURE_2D, camText);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 5*sizeof(GLfloat), camVBO);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(GLfloat), &(camVBO[3]));
	glDrawArrays(GL_QUADS, 0, 4);
	glDisableClientState(GL_VERTEX_ARRAY);	
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void display(void){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();					
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	drawCam();
	drawHilt();
	drawBlade();

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
