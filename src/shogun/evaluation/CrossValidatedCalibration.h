/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/machine/Machine.h>
#include <shogun/evaluation/Calibration.h>
#include <shogun/lib/config.h>


#ifndef _CROSS_VALIDATED_CALIBRATION_H__
#define _CROSS_VALIDATED_CALIBRATION_H__


namespace shogun 
{

class CCrossValidatedCalibration: public CMachine 
{

public:
	virtual const char* get_name() const
	{
		return "CrossValidatedCalibration";
	}
	
	virtual EProblemType get_machine_problem_type() const;

	virtual bool train(CFeatures* data=NULL);

	virtual CBinaryLabels* apply_binary(CFeatures* features=NULL);

	virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> subset_indices);

	virtual bool train_locked(SGVector<index_t> indices);

	virtual CMulticlassLabels* apply_locked_multiclass(SGVector<index_t> subset_indices);

	virtual CMulticlassLabels* apply_multiclass(CFeatures* features);

	CCrossValidatedCalibration();

	/** constructor
	 * @param machine learning machine to use
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param calibrator calibration machine to use
	 * @param autolock autolock
	 */
	CCrossValidatedCalibration(
	    CMachine* machine, CLabels* labels,
	    CSplittingStrategy* splitting_strategy, CCalibrationMethod* calibration_method);

	~CCrossValidatedCalibration();

	void init();

	/** get learning machine
	*/
	CMachine* get_machine() const;

	private:
		CDynamicObjectArray* m_calibration_machines;
		CMachine* m_machine;
		CLabels* m_labels;
		CSplittingStrategy* m_splitting_strategy;
		CCalibrationMethod* m_calibration_method;

};
}
#endif