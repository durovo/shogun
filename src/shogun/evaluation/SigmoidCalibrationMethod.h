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
#include <shogun/mathematics/Math.h>
#include <shogun/lib/config.h>
#include <shogun/evaluation/CalibrationMethod.h>

#ifndef _SIGMOID_CALIBRATION_METHOD_H__
#define _SIGMOID_CALIBRATION_METHOD_H__

namespace shogun 
{

class CSigmoidCalibrationMethod: public CCalibrationMethod
{
public:
	virtual const char* get_name() const
	{
		return "SigmoidCalibrationMethod";
	}

	virtual EProblemType get_machine_problem_type() const
	{
		return PT_BINARY;
	}

	virtual bool train(SGVector<float64_t> values);

	virtual SGVector<float64_t> apply_binary(SGVector<float64_t> values);

	void set_target_values(SGVector<float64_t> target_values);

	/** constructor, for use with custom kernels (no features)
	 * @param machine learning machine to use
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock autolock
	 */
	CSigmoidCalibrationMethod();

	CSigmoidCalibrationMethod(SGVector<float64_t> target_values);

	~CSigmoidCalibrationMethod();

private:
	float64_t a, b;
};
}
#endif