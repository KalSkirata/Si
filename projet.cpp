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

GLfloat face[]={-xmax, -ymax, 0.0,
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
		1.0, 0.0,
};

static void initGL                 (void);
static void reshape                (int, int);
static void displayQuads    (void);
static void display                (void);

static GLuint texture=0;

//CvCapture* capture;
VideoCapture *capture;

IplImage* frame;
Mat mat_frame;
CascadeClassifier* hand_cc;

Mat ci, gsi;

void initCV(){
	//capture = cvCaptureFromCAM(CV_CAP_ANY);
		
	//if( !hand_cc.load(xmlPath) ){ printf("--(!)Error loading xml file\n");};
	
	hand_cc = new CascadeClassifier(xmlPath);
	
	capture = new VideoCapture(CV_CAP_ANY);
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
		glGenerateMipmap(GL_TEXTURE_2D);
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

static void initGL(void){
	initCV();
	glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	/*glimageLoadAndBind("visages.jpg", &texture);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);*/

	glBindTexture(GL_TEXTURE_2D, 0);
}

void reshape(int width, int height){
	glViewport(0,0,(GLsizei)(width),(GLsizei)(height));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (GLfloat)width/height, 0.01f, 1000.0f);	
	glMatrixMode(GL_MODELVIEW);
}

static void displayQuads(void){
	/*frame = cvQueryFrame(capture);
	texture = ConvertIplToTexture(frame);*/
	
	//mat_frame = cvQueryFrame(capture);
	detectHand();
	texture = matToTexture(ci, GL_NEAREST, GL_NEAREST, GL_CLAMP);

	glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	glBindTexture(GL_TEXTURE_2D, texture);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 8*sizeof(GLfloat), face);
	//glColorPointer(3, GL_FLOAT, 8*sizeof(GLfloat), &(face[3]));
	glTexCoordPointer(2, GL_FLOAT, 8*sizeof(GLfloat), &(face[6]));

	glDrawArrays(GL_QUADS, 0, 4);

	//glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void display(void){
	static GLfloat rot = 0.0;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();					
	gluLookAt(0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	displayQuads();

	glutSwapBuffers();
	rot +=3.0;
}

static void keyboard(unsigned char key, int x, int y){
	if (key == 27 ) exit(0);
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
