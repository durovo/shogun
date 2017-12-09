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
#include <shogun/evaluation/CalibrationMethod.h>

using namespace shogun;

SGVector<float64_t> CCalibrationMethod::apply_binary(SGVector<float64_t> values) 
{
	SG_NOTIMPLEMENTED
	
}

bool CCalibrationMethod::train(SGVector<float64_t> values)
{
	SG_NOTIMPLEMENTED

	return true;
}

CCalibrationMethod::CCalibrationMethod(): CMachine(){}

void CCalibrationMethod::set_target_values(SGVector<float64_t> target_values)
{
	m_target_values = target_values;
}

CCalibrationMethod::CCalibrationMethod(SGVector<float64_t> target_values) 
{
	m_target_values = target_values;
}

CCalibrationMethod::~CCalibrationMethod() {}