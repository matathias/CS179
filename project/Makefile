###############################################################################
# CS/CNS 171 Fall 2014
#
# This is a template Makefile for OpenGL programs. Edit it however you find
# convenient.
# 
# The current version of this file should compile OpenGL programs just fine on
# Debian-based Linux operating systems.
#
# If you run Mac OS or other distributions of Linux, then you may have to
# fool around with the 'INCLUDE' and 'LIBDIR' lines below before the Makefile
# can compile the OpenGL parts successfully.
###############################################################################
CC = g++
FLAGS = -g -o

INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -I/home/warrick/Dropbox/r-dwarrick-cs171/hw7/files
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SOURCES1 = util.cpp raytrace.cpp
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME1 = raytrace

all: raytrace

raytrace: $(SOURCES1)
	$(CC) $(FLAGS) $(EXENAME1) $(INCLUDE) $(LIBDIR) $(SOURCES1) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean

