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
		return "Calibration";
	}
	
	virtual EProblemType get_machine_problem_type() const;

	virtual bool train(CFeatures* data=NULL);

	// bool get_probabilities(CFeatures* data=NULL);

	virtual CBinaryLabels* apply_binary(CFeatures* features=NULL);

	virtual CBinaryLabels* apply_locked_binary(SGVector<index_t> subset_indices);

	// virtual CMulticlassLabels* apply_multiclass(CFeatures* features=NULL);

	// virtual CMulticlassLabels* apply_multiclass_locked(SGVector<index_t> subset_indices);

	virtual bool train_locked(SGVector<index_t> indices);

	CCrossValidatedCalibration();

	/** constructor
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion evaluation criterion to use
	 * @param calibrator calibration machine to use
	 * evaluation
	 */
	CCrossValidatedCalibration(
	    CMachine* machine, CFeatures* features, CBinaryLabels* labels,
	    CSplittingStrategy* splitting_strategy, CCalibration* calibrator);

	/** constructor, for use with custom kernels (no features)
	 * @param machine learning machine to use
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param calibrator calibration machine to use
	 * @param autolock autolock
	 */
	CCrossValidatedCalibration(
	    CMachine* machine, CBinaryLabels* labels,
	    CSplittingStrategy* splitting_strategy, CCalibration* calibrator);

	~CCrossValidatedCalibration();

	void init();

	/** get learning machine
	*/
	CMachine* get_machine() const;

	private:
		CDynamicObjectArray* m_calibration_machines;
		CMachine* m_machine;
		CBinaryLabels* m_labels;
		CSplittingStrategy* m_splitting_strategy;
		CFeatures* m_features;
		CCalibration* m_calibrator;

};
}
#endif