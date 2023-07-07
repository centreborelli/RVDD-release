// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2012, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.


#ifndef BICUBIC_INTERPOLATION_C
#define BICUBIC_INTERPOLATION_C

#include <stdbool.h>

/**
  *
  * Neumann boundary condition test
  *
**/
static int neumann_bc(int x, int nx, bool *out);

/**
  *
  * Periodic boundary condition test
  *
**/
static int periodic_bc(int x, int nx, bool *out);


/**
  *
  * Symmetric boundary condition test
  *
**/
static int symmetric_bc(int x, int nx, bool *out);


/**
  *
  * Cubic interpolation in one dimension
  *
**/
static double cubic_interpolation_cell (
	double v[4],  //interpolation points
	double x      //point to be interpolated
);


/**
  *
  * Bicubic interpolation in two dimensions
  *
**/
static double bicubic_interpolation_cell (
	double p[4][4], //array containing the interpolation points
	double x,       //x position to be interpolated
	double y        //y position to be interpolated
);

/**
  *
  * Compute the bicubic interpolation of a point in an image.
  * Detect if the point goes outside the image domain.
  *
**/
float bicubic_interpolation_at(
	const float *input, //image to be interpolated
	const float  uu,    //x component of the vector field
	const float  vv,    //y component of the vector field
	const int    nx,    //image width
	const int    ny,    //image height
	bool         border_out //if true, return zero outside the region
);


/**
  *
  * Compute the bicubic interpolation of an image.
  *
**/
void bicubic_interpolation_warp(
	const float *input,     // image to be warped
	const float *u,         // x component of the vector field
	const float *v,         // y component of the vector field
	float       *output,    // image warped with bicubic interpolation
	const int    nx,        // image width
	const int    ny,        // image height
	bool         border_out // if true, put zeros outside the region
);


#endif//BICUBIC_INTERPOLATION_C
