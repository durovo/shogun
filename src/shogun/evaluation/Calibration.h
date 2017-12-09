/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/config.h>
#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/machine/Machine.h>

#ifndef _CALIBRATION_H__
#define _CALIBRATION_H__

namespace shogun 
{

class CCalibration: public CMachine 
{
public:
	virtual const char* get_name() const
	{
		return "Calibration";
	}

	virtual EProblemType get_machine_problem_type() const;

	virtual bool train(CFeatures* data=NULL);

	virtual bool train_locked(SGVector<index_t> subset_indices);

	virtual CBinaryLabels* apply_binary(CFeatures* features);

	virtual CMulticlassLabels* apply_multiclass(CFeatures* features);	

	virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> subset_indices);

	/** constructor
	 */
	CCalibration();

	~CCalibration();

	void init();

	void set_machine(CMachine* machine);

	void set_calibration_method(CCalibrationMethod* calibration_method);

	CMachine* get_machine();

private:
	CMachine* m_machine;
	CFeatures* m_features;
	float64_t a, b;
	CDynamicObjectArray* m_calibration_machines;
	CCalibrationMethod* m_method;

};
}
#endif