/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/evaluation/SigmoidCalibrationMethod.h>

using namespace shogun;

SGVector<float64_t> CSigmoidCalibrationMethod::apply_binary(SGVector<float64_t> values) 
{
	for (index_t i = 0; i < values.vlen; ++i)
	{
		float64_t fApB = values[i] * a + b;
		values[i] = fApB >= 0 ? CMath::exp(-fApB) / (1.0 + CMath::exp(-fApB)) :
		                      1.0 / (1 + CMath::exp(fApB));
	}
	return values;
	
}

bool CSigmoidCalibrationMethod::train(SGVector<float64_t> values)
{
	CStatistics::SigmoidParamters params =
		        CStatistics::fit_sigmoid(values);
	a = params.a;
	b = params.b;

	return true;
}

CSigmoidCalibrationMethod::CSigmoidCalibrationMethod(): CCalibrationMethod(){}

CSigmoidCalibrationMethod::CSigmoidCalibrationMethod(SGVector<float64_t> target_values):
CCalibrationMethod(target_values){};

CSigmoidCalibrationMethod::~CSigmoidCalibrationMethod() {}