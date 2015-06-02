/* David Warrick
 * raytraceCuda.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <math.h>
#define _USE_MATH_DEFINES
#include <float.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>
#include "raytraceCuda.cuh"

using namespace std;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

struct Point_Light
{
    double position[3];    //3-vector
    double color[3];       //3-vector
    double attenuation_k;
};

struct Material
{
    double diffuse[3];     //3-vector
    double ambient[3];     //3-vector
    double specular[3];    //3-vector
    double shine;
    double snell;
    double opacity;
};

struct Object
{
    double e;
    double n;
    Material mat;
    double scale[9];      //3x3-matrix
    double unScale[9];    //3x3-matrix
    double rotate[9];     //3x3-matrix
    double unRotate[9];   //3x3-matrix
    double translate[3];   //3-vector
    double unTranslate[3]; //3-vector
};

struct Pixel
{
    double red;
    double green;
    double blue;
};

/******************************************************************************/
// Global variables

// Tolerance value for the Newton's Method update rule
double epsilon = 0.00001;

// Toggle for using default object or objects loaded from input
bool defaultObject = true;
// Toggle for using default lights or lights loaded from input
bool defaultLights = true;
// Toggle for using antialiasing
bool antiAlias = false;

/* Ray-tracing globals */
// Unit orthogonal film vectors
double e1[] = {1.0, 0.0, 0.0};
double e2[] = {0.0, 1.0, 0.0};
double e3[] = {0.0, 0.0, 1.0};

double lookAt[] = {0.0, 0.0, 0.0};
double lookFrom[] = {5.0, 5.0, 5.0};

double up[] = {0.0, 1.0, 0.0};

double bgColor[] = {0.0, 0.0, 0.0};

double filmDepth = 0.05;
double filmX = 0.035;
double filmY = 0.035;
int Nx = 100, Ny = 100;

vector<Point_Light*> lightsPPM;
vector<Object*> objects;

Point_Light *p_lights;
Object *p_objects;

/******************************************************************************/
// Function prototypes
void initPPM();
void create_film_plane(double *e1, double *e2, double *e3);

void create_Material(double dr, double dg, double db, 
                     double ar, double ag, double ab,
                     double sr, double sg, double sb, 
                     double shine, double refract, double opac, 
                     Material *mat);
void create_default_material(Material *m);

void create_object(double e, double n, double xt, double yt, double zt, 
                   double a, double b, double c, double r1, double r2, 
                   double r3, double theta, Object *obj);
void change_object_material(Object *obj, Material *mat);
void create_default_object();

void create_Light(double x, double y, double z, double r, double g, double b,
                  double k, Point_Light *l);
void create_PPM_lights();

// Print pixel data to output
void printPPM(int pixelIntensity, int xre, int yre, double *grid);
// Function to parse the command line arguments
void parseArguments(int argc, char* argv[]);
void getArguments(int argc, char* argv[]);
void parseFile(char* filename);

/* Returns the norm of the given vector. */
double norm(double *vec);
/* Normalizes the given vector. */
void normalize(double *vec);
/* Returns the dot product of the given vectors. */
double dot(double *a, double *b);
/* Returns the cross product a x b into vector c. */
void cross(double *a, double *b, double *c);

/* Gets the rotation matrix for a given rotation axis (x, y, z) and angle. */
void get_rotate_mat(double x, double y, double z, double angle, double *m);
/* Gets the scaling matrix for a given scaling vector (x, y, z). */
void get_scale_mat(double x, double y, double z, double *m);

// Simple function to convert an angle in degrees to radians
double deg2rad(double angle);
// Simple function to convert an angle in radians to degrees
double rad2deg(double angle);

/******************************************************************************/
// Function declarations
/******************************************************************************/
// Helper functions

/* Returns the norm of the given vector. */
double norm(double *vec)
{
    double n = 0;
    for (int i = 0; i < 3; i++) {
        n += vec[i] * vec[i];
    }
    return sqrt(n);
}

/* Normalizes the given vector. */
void normalize(double *vec)
{
    double n = norm(vec);
    for (int i = 0; i < 3; i++) {
        vec[i] = vec[i] / n;
    }
}

/* Returns the dot product of the given vectors. */
double dot(double *a, double *b)
{
    double d = 0;
    for (int i = 0; i < 3; i++) {
        d += a[i] * b[i];
    }
    return d;
}

/* Returns the cross product a x b into vector c. */
void cross(double *a, double *b, double *c)
{
    c[0] = (a[1] * b[2]) - (a[2] * b[1]);
    c[1] = (a[2] * b[0]) - (a[0] * b[2]);
    c[2] = (a[0] * b[1]) - (a[1] * b[0]);
}

/* Gets the rotation matrix for a given rotation axis (x, y, z) and angle. */
void get_rotate_mat(double x, double y, double z, double angle, double *m)
{
    double nor = sqrt((x * x) + (y * y) + (z * z));
    x = x / nor;
    y = y / nor;
    z = z / nor;
    angle = deg2rad(angle);

    m[0] = pow(x,2) + (1 - pow(x,2)) * cos(angle);
    m[1] = (x * y * (1 - cos(angle))) - (z * sin(angle));
    m[2] = (x * z * (1 - cos(angle))) + (y * sin(angle));

    m[3] = (y * x * (1 - cos(angle))) + (z * sin(angle));
    m[4] = pow(y,2) + (1 - pow(y,2)) * cos(angle);
    m[5] = (y * z * (1 - cos(angle))) - (x * sin(angle));

    m[6] = (z * x * (1 - cos(angle))) - (y * sin(angle));
    m[7] = (z * y * (1 - cos(angle))) + (x * sin(angle));
    m[8] = pow(z,2) + (1 - pow(z,2)) * cos(angle);
}

/* Gets the scaling matrix for a given scaling vector (x, y, z). */
void get_scale_mat(double x, double y, double z, double *m)
{
    m[0] = x;
    m[1] = 0;
    m[2] = 0;
    
    m[3] = 0;
    m[4] = y;
    m[5] = 0;
    
    m[6] = 0;
    m[7] = 0;
    m[8] = z;
}

// Simple function to convert an angle in degrees to radians
double deg2rad(double angle)
{
    return angle * M_PI / 180.0;
}
// Simple function to convert an angle in radians to degrees
double rad2deg(double angle)
{
    return angle * (180.0 / M_PI);
}

/******************************************************************************/
// Raytracing functions

void initPPM()
{
    if (defaultObject)
        create_default_object();
    if (defaultLights)
        create_PPM_lights();
        
    create_film_plane(&e1[0], &e2[0], &e3[0]);
}


void create_film_plane(double *e1, double *e2, double *e3)
{
    // First, find the proper value for filmY from filmX, Nx and Ny
    filmY = Ny * filmX / (double) Nx;
    
    // Find and set the plane vectors
    for (int i = 0; i < 3; i++) {
        e3[i] = lookAt[i] - lookFrom[i];
    }
    normalize(e3);
    
    double alpha = dot(up, e3) / (double) dot(e3, e3);
    for (int i = 0; i < 3; i++) {
        e2[i] = up[i] - (alpha * e3[i]);
    }
    normalize(e2);
    
    cross(e2, e3, e1);
    normalize(e1);
}

void create_Material(double dr, double dg, double db, 
                     double ar, double ag, double ab,
                     double sr, double sg, double sb, 
                     double shine, double refract, double opac, 
                     Material *mat)
{
    mat->diffuse[0] = dr;
    mat->diffuse[1] = dg;
    mat->diffuse[2] = db;
    mat->ambient[0] = ar;
    mat->ambient[0] = ag;
    mat->ambient[0] = ab;
    mat->specular[0] = sr;
    mat->specular[1] = sg;
    mat->specular[2] = sb;
    
    mat->shine = shine;
    mat->snell = refract;
    mat->opacity = opac;
}

void create_default_material(Material *m)
{
    create_Material(0.5, 0.5, 0.5,      // diffuse rgb
                    0.01, 0.01, 0.01,   // ambient rgb
                    0.5, 0.5, 0.5,      // specular rgb
                    10, 0.9, 0.01, m);
}

void create_object(double e, double n, double xt, double yt, double zt, 
                   double a, double b, double c, double r1, double r2, 
                   double r3, double theta, Object *obj)
{
    obj->e = e;
    obj->n = n;
    
    get_scale_mat(a, b, c, &obj->scale[0]);
    get_scale_mat(1 / (double) a, 1 / (double) b, 1 / (double) c, &obj->unScale[0]);
    get_rotate_mat(r1, r2, r3, theta, &obj->rotate[0]);
    get_rotate_mat(r1, r2, r3, -theta, &obj->unRotate[0]);
    
    obj->translate[0] = xt;
    obj->translate[1] = yt;
    obj->translate[2] = zt;
    obj->unTranslate[0] = -xt;
    obj->unTranslate[1] = -yt;
    obj->unTranslate[2] = -zt;

    create_default_material(&obj->mat);
}

void change_object_material(Object *obj, Material *mat)
{
    for(int i = 0; i < 3; i++) {
        obj->mat.diffuse[i] = mat->diffuse[i];
        obj->mat.ambient[i] = mat->ambient[i];
        obj->mat.specular[i] = mat->specular[i];
    }
    obj->mat.shine = mat->shine;
    obj->mat.snell = mat->snell;
    obj->mat.opacity = mat->opacity;
}

void create_default_object()
{
    Object *obj = (Object *)malloc(sizeof(Object));
    create_object(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, obj);

    objects.push_back(obj);
}

void create_Light(double x, double y, double z, double r, double g, double b,
                  double k, Point_Light *l)
{    
    l->position[0] = x;
    l->position[1] = y;
    l->position[2] = z;
    
    l->color[0] = r;
    l->color[1] = g;
    l->color[2] = b;
    
    l->attenuation_k = k;
}

void create_PPM_lights()
{
    Point_Light *eyeLight = (Point_Light *)malloc(sizeof(Point_Light));
    create_Light(lookFrom[0], lookFrom[1], lookFrom[2], 1, 1, 1, 0.01, eyeLight);
    lightsPPM.push_back(eyeLight);

    Point_Light *light1 = (Point_Light *)malloc(sizeof(Point_Light));
    create_Light(-10, 10, 10, 1, 1, 1, 0.001, light1);
    lightsPPM.push_back(light1);
}

// Function to parse the command line arguments
void parseArguments(int argc, char* argv[])
{
    printf("entered parseArguments\n");
    int inInd = 1;
    
    // Command line triggers to respond to.
    const char* objectsIn = "-obj"; // the following values are: e, n, xt, yt, zt,
                                   // a, b, c, r1, r2, r3, theta
    const char* inMats = "-mat"; // the next several values are the diffuse rgb, 
                                  //specular rgb, shininess value, and refractive index of the 
                                  // object material
    const char* epsilonC = "-ep"; // the epsilon-close-to-zero value for the update rule
    const char* background = "-bg"; // next three values are the rgb for the background
    const char* target = "-la"; // the next three values are the x,y,z for the look at vector
    const char* eye = "-eye"; // the next three values are the x,y,z for the look from vector
    const char* filmP = "-f"; // the next two values are the film plane depth and film plane width
    const char* nres = "-res"; // the next two valures are Nx and Ny
    const char* filmUp = "-up"; // the next three values are the x,y,z forr the film plane "up" vector
    const char* inLights = "-l"; // the next several values are the position, 
                                 //color, and attenuation coefficient for a new light
    const char* inEye = "-le"; // the next four values are the rgb and k for the eye light.
                                // only one eye light can be specified.
    const char* antiAliasing = "-anti"; // toggles antialiasing

    // Temporary values to store the in parameters. These only get assigned to
    // the actual program values if no errors are encountered while parsing the
    // arguments.
    vector<Object*> tempObjs;
    vector<Material*> tempMats;
    vector<Point_Light*> tempLights;

    double tepsilon = epsilon;
    double tlookAt[3], tlookFrom[3], tup[3], tbgColor[3];
    for (int i = 0; i < 3; i++) {
        tlookAt[i] = lookAt[i];
        tlookFrom[i] = lookFrom[i];
        tup[i] = up[i];
        tbgColor[i] = bgColor[i];
    }
    double tfilmDepth = filmDepth;
    double tfilmX = filmX;
    int tNx = Nx, tNy = Ny;

    double eyeColor[] = {1.0,1.0,1.0};
    double eyeK = 0.01;

    bool tdefaultObject = defaultObject;
    bool tdefaultLights = defaultLights;
    bool eyeSpecified = false;
    bool tantiAlias = antiAlias;

    try
    {
        printf("entered try block\n");
        while (inInd < argc)
        {
            printf("start of loop\n");
            if (strcmp(argv[inInd], objectsIn) == 0)
            {
                inInd += 12;
                if (inInd >= argc) 
                    throw out_of_range("Missing argument(s) for -obj [e n xt yt zt a b c r1 r2 r3 theta]");
                Object *tobj = (Object *)malloc(sizeof(Object));
                create_object(atof(argv[inInd-11]), atof(argv[inInd-10]),
                              atof(argv[inInd-9]), atof(argv[inInd-8]),
                              atof(argv[inInd-7]), atof(argv[inInd-6]),
                              atof(argv[inInd-5]), atof(argv[inInd-4]),
                              atof(argv[inInd-3]), atof(argv[inInd-2]),
                              atof(argv[inInd-1]), atof(argv[inInd]), tobj);
                tempObjs.push_back(tobj);
                tdefaultObject = false;
            }
            else if (strcmp(argv[inInd], inMats) == 0)
            {
                inInd += 9;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -mat [dr dg db sr sg sb shine refraction opacity]");
                tdefaultObject = false;
                Material *mat = (Material*)malloc(sizeof(Material));
                create_Material(atof(argv[inInd-8]), atof(argv[inInd-7]),
                                atof(argv[inInd-6]), 0, 0, 0, 
                                atof(argv[inInd-5]), atof(argv[inInd-4]),
                                atof(argv[inInd-3]), atof(argv[inInd-2]),
                                atof(argv[inInd-1]), atof(argv[inInd]), mat);
                tempMats.push_back(mat);
            }
            else if (strcmp(argv[inInd], inLights) == 0)
            {
                inInd += 7;
                if (inInd >= argc) throw 
                                out_of_range("Missing argument(s) for -l [x y z r g b k]");
                tdefaultLights = false;
                Point_Light *light = (Point_Light *)malloc(sizeof(Point_Light));
                create_Light(atof(argv[inInd-6]), atof(argv[inInd-5]),
                             atof(argv[inInd-4]), atof(argv[inInd-3]),
                             atof(argv[inInd-2]), atof(argv[inInd-1]),
                             atof(argv[inInd]), light);
                tempLights.push_back(light);
            }
            else if (strcmp(argv[inInd], inEye) == 0)
            {
                inInd += 4;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -le [r g b k]");
                if (!eyeSpecified)
                {
                    eyeColor[0] = atof(argv[inInd-3]);
                    eyeColor[1] = atof(argv[inInd-2]);
                    eyeColor[2] = atof(argv[inInd-1]);
                    eyeK = atof(argv[inInd]);
                    eyeSpecified = true;
                    tdefaultLights = false;
                }
            }
            else if (strcmp(argv[inInd], nres) == 0)
            {
                inInd += 2;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -res [Nx Ny]");
                tNx = atof(argv[inInd-1]);
                tNy = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], filmP) == 0)
            {
                inInd += 2;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -f [Fd Fx]");
                tfilmDepth = atof(argv[inInd-1]);
                tfilmX = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], epsilonC) == 0)
            {
                inInd++;
                if (inInd >= argc) throw out_of_range("Missing argument for -ep [epsilon]");
                tepsilon = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], background) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -bg [x y z]");
                tbgColor[0] = atof(argv[inInd-2]);
                tbgColor[1] = atof(argv[inInd-1]);
                tbgColor[2] = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], target) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -la [x y z]");
                tlookAt[0] = atof(argv[inInd-2]);
                tlookAt[1] = atof(argv[inInd-1]);
                tlookAt[2] = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], eye) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -eye [x y z]");
                tlookFrom[0] = atof(argv[inInd-2]);
                tlookFrom[1] = atof(argv[inInd-1]);
                tlookFrom[2] = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], filmUp) == 0)
            {
                inInd += 3;
                if (inInd >= argc) throw out_of_range("Missing argument(s) for -up [x y z]");
                tup[0] = atof(argv[inInd-2]);
                tup[1] = atof(argv[inInd-1]);
                tup[2] = atof(argv[inInd]);
            }
            else if (strcmp(argv[inInd], antiAliasing) == 0)
            {
                tantiAlias = true;
            }

            inInd++;
        }

        epsilon = tepsilon;
        filmDepth = tfilmDepth, filmX = tfilmX;
        printf("Assigning Nx and Ny as tNx: %d and tNy: %d\n", tNx, tNy);
        Nx = tNx, Ny = tNy;
        defaultLights = tdefaultLights;
        defaultObject = tdefaultObject;
        antiAlias = tantiAlias;
        for (int i = 0; i < 3; i++) {
            lookAt[i] = tlookAt[i];
            lookFrom[i] = tlookFrom[i];
            up[i] = tup[i];
            bgColor[i] = tbgColor[i];
        }

        unsigned int i = 0;
        while (i < tempMats.size() &&  i < tempObjs.size())
        {
            change_object_material(tempObjs[i], tempMats[i]);
            i++;
        }

        objects = tempObjs;

        Point_Light *eye = (Point_Light *)malloc(sizeof(Point_Light));
        create_Light(lookFrom[0], lookFrom[1], lookFrom[2], eyeColor[0],
                     eyeColor[1], eyeColor[2], eyeK, eye);

        vector<Point_Light*>::iterator it = tempLights.begin();

        tempLights.insert(it, eye);

        if (!defaultLights)
        {
            lightsPPM = tempLights;
        }
    }
    catch (exception& ex)
    {
        cout << "Error at input argument " << inInd << ":" << endl;
        cout << ex.what() << endl;
        cout << "Program will execute with default values." << endl;
    }
}

void getArguments(int argc, char* argv[])
{
    if (argc > 1)
    {
        string filetype = ".txt";
        string firstArg(argv[1]);
        unsigned int isFile = firstArg.find(filetype);
        if (isFile != string::npos)
        {
            parseFile(argv[1]);
        }
        else
        {
            parseArguments(argc, argv);
        }
    }
}

void parseFile(char* filename)
{
    ifstream ifs;
    ifs.open(filename);

    vector<char* > input;

    // Retrieve the data
    while(ifs.good())
    {
        // Read the next line
        string nextLine;
        getline(ifs, nextLine);

        while (nextLine.length() > 0)
        {
            // Get rid of extra spaces and read in any numbers that are
            // encountered
            string rotStr = " ";
            while (nextLine.length() > 0 && rotStr.compare(" ") == 0)
            {
                int space = nextLine.find(" ");
                if (space == 0)
                    space = 1;
                rotStr = nextLine.substr(0, space);
                nextLine.erase(0, space);
            }
            char* thistr = new char[rotStr.length() + 1];
            strcpy(thistr, rotStr.c_str());
            input.push_back(thistr);
        }
    }
    ifs.close();

    char* args[input.size()+1];
    for (unsigned int i = 0; i < input.size(); i++)
        args[i+1] = input.at(i);

    parseArguments(input.size()+1, args);
}

// Print pixel data to output
void printPPM(int pixelIntensity, int xre, int yre, double *grid)
{
    // Print the PPM data to standard output
    cout << "P3" << endl;
    cout << xre << " " << yre << endl;
    cout << pixelIntensity << endl;

    for (int j = 0; j < yre; j++)
    {
        for (int i = 0; i < xre; i++)
        {
            int index = (j * xre + i) * 3;
            int red = grid[index] * pixelIntensity;
            int green = grid[index + 1] * pixelIntensity;
            int blue = grid[index + 2] * pixelIntensity;
            
            cout << red << " " << green << " " << blue << endl;
        }
    }
}

int main(int argc, char* argv[])
{
    // extract the command line arguments
    getArguments(argc, argv);
    
    // block size will be 16 x 16 = 2^4 x 2^4
    int blockPower = 4;

    initPPM();
    /***** Allocate memory here *****/    
    double *grid = (double*)malloc(sizeof(double) * Ny * Nx * 3);
    
    int numObjects = objects.size();
    p_objects = (Object *)malloc(sizeof(Object) * numObjects);
    for (int j = 0; j < numObjects; j++) {
        p_objects[j] = *objects[j];
    }
    int numLights = lightsPPM.size();
    p_lights = (Point_Light *)malloc(sizeof(Point_Light) * numLights);
    for (int j = 0; j < numLights; j++) {
        p_lights[j] = *lightsPPM[j];
    }
    
    /* Allocate memory on the GPU */
    double *d_e1, *d_e2, *d_e3, *d_lookFrom, *d_up, *d_bgColor;
    double *d_grid;
    gpuErrChk(cudaMalloc(&d_e1, 3 * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_e2, 3 * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_e3, 3 * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_lookFrom, 3 * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_up, 3 * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_bgColor, 3 * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_grid, sizeof(double) * Ny * Nx * 3));
    
    /* Copy data from the cpu to the gpu. */
    gpuErrChk(cudaMemcpy(d_e1, &e1[0], 3 * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_e2, &e2[0], 3 * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_e3, &e3[0], 3 * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_lookFrom, &lookFrom[0], 3 * sizeof(double), 
                         cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_up, &up[0], 3 * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_bgColor, &bgColor[0], 3 * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    gpuErrChk(cudaMemset(d_grid, 0, sizeof(double) * Ny * Nx * 3));
    
    /* Handle the allocating and copying of the Objects and Point_Lights arrays.
     * This is a little weird because the structs store pointers...
     */
    //int numObjects = objects.size();
    //int numLights = lightsPPM.size();
    Object *d_objects;
    Point_Light *d_lights;
    gpuErrChk(cudaMalloc(&d_objects, numObjects * sizeof(Object)));
    gpuErrChk(cudaMalloc(&d_lights, numLights * sizeof(Object)));
    // Copy the objects onto the gpu, as well as allocating space for the object
    // pointers and copying the data in there
    /*for (int i = 0; i < numObjects; i++)
    {
        gpuErrChk(cudaMemcpy(&d_objects[i], objects[i], sizeof(Object), 
                             cudaMemcpyHostToDevice));
        
    }*/
    gpuErrChk(cudaMemcpy(d_objects, p_objects, sizeof(Object) * numObjects,
                         cudaMemcpyHostToDevice));
    // Do the same for the Point_Lights
    /*for (int i = 0; i < numLights; i++)
    {
        gpuErrChk(cudaMemcpy(&d_lights[i], lightsPPM[i], sizeof(Point_Light), 
                             cudaMemcpyHostToDevice));
    }*/
    gpuErrChk(cudaMemcpy(d_lights, p_lights, sizeof(Point_Light) * numLights,
                         cudaMemcpyHostToDevice));
    
    /* Call the GPU code. */
    callRaytraceKernel(d_grid, d_objects, numObjects, d_lights, numLights,
                       Nx, Ny, filmX, filmY, d_bgColor, d_e1, d_e2, d_e3,
                       d_lookFrom, epsilon, filmDepth, antiAlias, blockPower);
    
    /* Copy data back to CPU. */
    gpuErrChk(cudaMemcpy(grid, d_grid, sizeof(double) * Ny * Nx * 3, 
                         cudaMemcpyDeviceToHost));

    /* Output the relevant data. */
    printPPM(255, Nx, Ny, grid);
    
    /* Free everything. */
    free(grid);
    free(p_objects);
    free(p_lights);
    
    gpuErrChk(cudaFree(d_e1));
    gpuErrChk(cudaFree(d_e2));
    gpuErrChk(cudaFree(d_e3));
    gpuErrChk(cudaFree(d_lookFrom));
    gpuErrChk(cudaFree(d_up));
    gpuErrChk(cudaFree(d_bgColor));
    gpuErrChk(cudaFree(d_grid));
    gpuErrChk(cudaFree(d_objects));
    gpuErrChk(cudaFree(d_lights));
}

