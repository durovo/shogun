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
#include <shogun/lib/config.h>

#ifndef _CALIBRATION_METHOD_H__
#define _CALIBRATION_METHOD_H__

namespace shogun 
{

class CCalibrationMethod: public CMachine 
{
public:
	virtual const char* get_name() const
	{
		return "CalibrationMethod";
	}

	virtual EProblemType get_machine_problem_type() const
	{
		return PT_BINARY;
	}

	virtual bool train(SGVector<float64_t> values);

	virtual SGVector<float64_t> apply_binary(SGVector<float64_t> values);

	void set_target_values(SGVector<float64_t> target_values);

	CCalibrationMethod();

	/** constructor
	 * @param target_values, value true value of each label
	 */
	CCalibrationMethod(SGVector<float64_t> target_values);

	~CCalibrationMethod();

private:
	SGVector<float64_t> m_target_values;
};
}
#endif