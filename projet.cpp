/** 
	\brief Remplace un tube de colle par un sabre laser sur une prise de caméra
	\author Lecocq Guillaume
	\file projet.cpp
**/

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

#include"glimage/glimage.hpp"

#define  GL_GLEXT_PROTOTYPES

/**
	\def définie si on affiche les différentes valeurs sur le terminal
**/
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

static /** identifiant de la texture de la caméra **/ GLuint camText=0, 
		 /** identifiant de la texture du manche du sabre laser **/ hiltText = 0,
		 /** identifiant de la texture de la lame du sabre laser **/bladeText = 0;

/** Capture de la caméra **/
VideoCapture *capture;

/** Matrice qui contiendra l'image original de la caméra **/
Mat imgOriginal;

/** Coordonnées (dans l'espace et de texture) du cube où la caméra sera texturée **/
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

GLfloat /** Coordonnées (dans l'espace et de texture) du cube où le manche sera texturé **/ hiltVBO[20], 
		  /** Coordonnées (dans l'espace et de texture) du cube où la lame sera texturée **/ bladeVBO[32];

/**
	\func initialise les paramètres relatifs à OpenCV
**/
void initCV(){
	capture = new VideoCapture(CV_CAP_ANY);
}

/**
	\func initialise les positions de l'épée dans l'espace à l'endroit où l'on détecte le tube de colle
	\param x coordonnées en x
	\param y coordonnées en y
	\param z coordonnées en z
	\param width largeur
	\param height hauteur
**/
void initSwordPosition(double x, double y, double z, double width, double height){
	//if(height > width){
		hiltVBO[0] = x+0.2; hiltVBO[1] = y; hiltVBO[2] = z;
		hiltVBO[3] = 0.0; hiltVBO[4] = 1.0;

		hiltVBO[5] = x+width+0.2; hiltVBO[6] = y; hiltVBO[7] = z;
		hiltVBO[8] = 0.0; hiltVBO[9] = 0.0;

		hiltVBO[10] = x+width+0.2;	hiltVBO[11] = y+height;	hiltVBO[12] = z;
		hiltVBO[13] = 1.0;	hiltVBO[14] = 0.0;

		hiltVBO[15] = x+0.2;	hiltVBO[16] = y+height;	hiltVBO[17] = z;
		hiltVBO[18] = 1.0;	hiltVBO[19] = 1.0;

	
		bladeVBO[0] = x+0.2; bladeVBO[1] = y+height-0.1; bladeVBO[2] = z;
		bladeVBO[3] = 0.0; bladeVBO[4] = 0.0;

		bladeVBO[5] = x+width+0.2; bladeVBO[6] = y+height-0.1; bladeVBO[7] = z;
		bladeVBO[8] = 1.0; bladeVBO[9] = 0.0;

		bladeVBO[10] = x+width+0.2;	bladeVBO[11] = y+4*height;	bladeVBO[12] = z;
		bladeVBO[13] = 1.0;	bladeVBO[14] = 1.0;

		bladeVBO[15] = x+0.2;	bladeVBO[16] = y+4*height;	bladeVBO[17] = z;
		bladeVBO[18] = 0.0;	bladeVBO[19] = 1.0;
	//}

	INFO("hiltVBO : (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) \n", hiltVBO[0], hiltVBO[1], hiltVBO[2], hiltVBO[5], hiltVBO[6], hiltVBO[7], hiltVBO[10], hiltVBO[11], hiltVBO[12], hiltVBO[15], hiltVBO[16], hiltVBO[17]);

	INFO("bladeVBO : (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) \n", bladeVBO[0], bladeVBO[1], bladeVBO[2], bladeVBO[5], bladeVBO[6], bladeVBO[7], bladeVBO[10], bladeVBO[11], bladeVBO[12], hiltVBO[15], hiltVBO[16], hiltVBO[17]);
}

/**
	\func transforme une matrice (ici l'image de la caméra) en une texture 2D
	\param mat adresse de la matrice
	\return ID de la texture 2D de la matrice
**/
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

/**
	\func convertit la coordonnée X de l'espace OpenCV vers l'espace OpengGL
	\param x coordonnée à convertir
	\param width largeur du rectangle
	\return coordonnée convertie
**/
double convertX(double x, double width){
	/*double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	INFO("convert x = %f \n", ((((2*x*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width);
	return ((((2*x*(xmax_cam/ymax_cam))/xmax_cam)-1)*(-1))-width;*/

	INFO("convert x = %f \n", ((((2*x*((double)imgOriginal.rows/(double)imgOriginal.cols))/(double)imgOriginal.rows)-1)*(-1))-width);
	return ((((2*x*((double)imgOriginal.rows/(double)imgOriginal.cols))/(double)imgOriginal.rows)-1)*(-1))-width;
}

/**
	\func convertit la coordonnée Y de l'espace OpenCV vers l'espace OpengGL
	\param y coordonnée à convertir
	\param height hauteur du rectangle
	\return coordonnée convertie
**/
double convertY(double y, double height){
	/*double xmax_cam = imgOriginal.rows, ymax_cam = imgOriginal.cols;
	INFO("convert y = %f \n", (((2*(ymax_cam-y*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height);
	return (((2*(ymax_cam-y*(ymax_cam/xmax_cam)))/ymax_cam)-1)-height;*/

	INFO("convert y = %f \n", (((2*((double)imgOriginal.cols-y*((double)imgOriginal.cols/(double)imgOriginal.rows)))/(double)imgOriginal.cols)-1)-height);
	return (((2*((double)imgOriginal.cols-y*((double)imgOriginal.cols/(double)imgOriginal.rows)))/(double)imgOriginal.cols)-1)-height;
}

/**
	\func convertit la largeur de l'espace OpenCV vers l'espace OpengGL
	\param width largeur à convertir
	\return largeur convertie
**/
double convertWidth(double width){
	return ((2*width)/imgOriginal.rows)+0.1;
}

/**
	\func convertit la hauteur de l'espace OpenCV vers l'espace OpengGL
	\param height hauteur à convertir
	\return hauteur convertie
**/
double convertHeight(double height){
	return ((2*height)/imgOriginal.cols)+0.1;
}

/**
	\func détecte la couleur jaune sur les images de la caméra (l'objet le plus gros est retourné sous la forme d'un rectangle (x,y,w,h))
**/
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
	initSwordPosition(convertX(bounding_rect.x,convertWidth(bounding_rect.width)+0.2),
							convertY(bounding_rect.y,convertHeight(bounding_rect.height)+0.2), 
							0.1,
							convertWidth(bounding_rect.width)+0.2,
							convertHeight(bounding_rect.height)+0.2);
}

/**
	\func initialise tous les paramètres relatifs à OpenGL
**/
static void initGL(void){
	glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	//for the transparency of the textures
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glimageLoadAndBind("img/hilt.png", &hiltText);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glimageLoadAndBind("img/blade.png", &bladeText);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glBindTexture(GL_TEXTURE_2D, 0);
}

/**
	\func redimensionne la fenêtre
	\param width largeur de la fenêtre
	\param height hauteur de la fenêtre
**/
void reshape(int width, int height){
	glViewport(0,0,(GLsizei)(width),(GLsizei)(height));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (GLfloat)width/height, 0.01f, 1000.0f);	
	glMatrixMode(GL_MODELVIEW);
}

/**
	\func dessine le manche du sabre laser sur un cube texturé
**/
static void drawHilt(){
	glBindTexture(GL_TEXTURE_2D, hiltText);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 5*sizeof(GLfloat), hiltVBO);
	glTexCoordPointer(2, GL_FLOAT, 5*sizeof(GLfloat), &(hiltVBO[3]));
	
	glDrawArrays(GL_QUADS, 0, 4);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);	
}

/**
	\func dessine la lame du sabre laser sur un cube texturé
**/
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

/**
	dessine la capture de la caméra sur un cube texturé
**/
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

/**
	\func dessine l'univers OpenGL
**/
void display(void){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();					
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	drawCam();
	drawHilt();
	drawBlade();

	glutSwapBuffers();
}

/**
	\n gère les entreés claviers
**/
static void keyboard(unsigned char key, int x, int y){
	switch(key){
		case 27: exit(0); break;
	}
}

int main(int argc, char **argv){
	glutInit(&argc, argv); 
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (500, 500); 
	glutInitWindowPosition (100, 100); 
	glutCreateWindow (argv[0]);
	initCV();
	initGL();  
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutIdleFunc(display);
	glutKeyboardFunc(keyboard);
	glutMainLoop(); 
	return 0; 
}
