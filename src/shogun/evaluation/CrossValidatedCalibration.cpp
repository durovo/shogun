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
#include <shogun/mathematics/Statistics.h>
#include <shogun/evaluation/CrossValidatedCalibration.h>
#include <shogun/evaluation/Calibration.h>
#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/lib/config.h>

using namespace shogun;

EProblemType CCrossValidatedCalibration::get_machine_problem_type() const 
{
	return m_machine->get_machine_problem_type();
}

bool CCrossValidatedCalibration::train(CFeatures* data)
{
	index_t num_subsets = m_splitting_strategy->get_num_subsets();

	m_calibration_machines = new CDynamicObjectArray(num_subsets);

	SG_DEBUG("building index sets for %d-fold cross-validated calibration\n", num_subsets)

	/* build index sets */
	m_splitting_strategy->build_subsets();

	if (m_machine->is_data_locked()) {
		SG_ERROR("cannot run train on locked data use train_locked instead\n")
		return false;

	}
	SG_DEBUG("starting unlocked calibration\n", get_name())
	/* tell machine to store model internally
	 * (otherwise changing subset of features will kaboom the classifier) */
	m_machine->set_store_model_features(true);

	for (index_t i = 0; i < num_subsets; ++i)
	{
		CMachine* machine;
		CFeatures* features;
		CBinaryLabels* labels;

		if (get_global_parallel()->get_num_threads() == 1)
		{
			machine = m_machine;
			features = data;
		}
		else
		{
			machine = (CMachine*)m_machine->clone();
			features = (CFeatures*)data->clone();
		}

		/* set feature subset for training */
		SGVector<index_t> inverse_subset_indices =
		    m_splitting_strategy->generate_subset_inverse(i);

		features->add_subset(inverse_subset_indices);

		/* set label subset for training */
		if (get_global_parallel()->get_num_threads() == 1)
			labels = (CBinaryLabels*)m_labels;
		else
			labels = (CBinaryLabels*)machine->get_labels();
		labels->add_subset(inverse_subset_indices);

		SG_DEBUG("training set %d:\n", i)
		if (io->get_loglevel() == MSG_DEBUG)
		{
			SGVector<index_t>::display_vector(
			    inverse_subset_indices.vector, inverse_subset_indices.vlen,
			    "training indices");
		}

		/* train machine on training features and remove subset */
		SG_DEBUG("starting training\n")
		machine->train(features);
		SG_DEBUG("finished training\n")
		features->remove_subset();
		labels->remove_subset();

		SGVector<index_t> subset_indices =
		    m_splitting_strategy->generate_subset_indices(i);
		features->add_subset(subset_indices);

		/* set label subset for testing */
		labels->add_subset(subset_indices);

		SG_DEBUG("test set %d:\n", i)
		if (io->get_loglevel() == MSG_DEBUG)
		{
			SGVector<index_t>::display_vector(
			    subset_indices.vector, subset_indices.vlen, "test indices");
		}

		CCalibration* calibrator = new CCalibration();
		calibrator->set_machine((CMachine*)machine->clone());
		calibrator->set_labels((CBinaryLabels*)labels->clone());
		calibrator->set_calibration_method((CCalibrationMethod*)m_calibration_method->clone());
		bool trained = calibrator->train((CFeatures*)features->clone());

		if (!trained) {
			return false;
		}

		m_calibration_machines->set_element(calibrator, i);

		
		SG_DEBUG("finished evaluation\n")
		features->remove_subset();

		/* clean up, remove subsets */
		labels->remove_subset();
		if (get_global_parallel()->get_num_threads() != 1)
		{
			SG_UNREF(machine);
			SG_UNREF(features);
			SG_UNREF(labels);
		}


	}

	SG_DEBUG("done unlocked evaluation\n", get_name())
	return true;

}

bool CCrossValidatedCalibration::train_locked(SGVector<index_t> indices)
{

	index_t num_subsets = m_splitting_strategy->get_num_subsets();

	m_calibration_machines = new CDynamicObjectArray(num_subsets);


	SG_DEBUG("building index sets for %d-fold cross-validated calibration\n", num_subsets)

	/* build index sets */
	m_splitting_strategy->build_subsets();

	if (m_machine->is_data_locked()) {
		SG_ERROR("please lock the data before running train_locked")
		return false;
	}

	SG_DEBUG("starting unlocked evaluation\n", get_name())
	/* tell machine to store model internally
	 * (otherwise changing subset of features will kaboom the classifier) */
	m_machine->set_store_model_features(true);

	for (index_t i = 0; i < num_subsets; ++i)
	{
		/* index subset for training, will be freed below */
		SGVector<index_t> inverse_subset_indices =
		    m_splitting_strategy->generate_subset_inverse(i);

		/* train machine on training features */
		m_machine->train_locked(inverse_subset_indices);

		/* feature subset for calibration */
		SGVector<index_t> subset_indices =
		    m_splitting_strategy->generate_subset_indices(i);

		/* set subset for testing labels */
		m_labels->add_subset(subset_indices);

		/* produce output for desired indices */
		CCalibration* calibrator = new CCalibration();
		calibrator->set_machine((CMachine*)m_machine->clone());
		calibrator->set_labels((CBinaryLabels*)m_labels->clone());
		calibrator->set_calibration_method((CCalibrationMethod*)m_calibration_method->clone());

		
		bool trained = calibrator->train_locked(subset_indices);

		if (!trained) {
			return false;
		}

		m_calibration_machines->set_element(calibrator, i);

		/* remove subset to prevent side effects */
		m_labels->remove_subset();

		SG_DEBUG("done locked evaluation\n", get_name())
	}
	return true;
}

CBinaryLabels* CCrossValidatedCalibration::apply_binary(CFeatures* features) 
{
	index_t num_machines = m_calibration_machines->get_num_elements();

	CBinaryLabels* temp_result;
	CMachine* temp_machine;
	SGVector<float64_t> result_values;
	CBinaryLabels* result_labels;

	for (index_t i = 0; i < num_machines; ++i) 
	{
		temp_machine = (CMachine*)m_calibration_machines->get_element(i);
		temp_result = (CBinaryLabels*)temp_machine->apply(features);
		if (i==0) 
		{
			result_values = temp_result->get_values();
			result_labels = temp_result;
		} 
		else 
		{
			result_values += temp_result->get_values();
		}
	}

	#pragma omp parallel for
	for (index_t i=0; i< result_values.vlen; ++i)
	{
		result_values[i] = result_values[i]/num_machines;
	}

	result_labels->set_values(result_values);

	return result_labels;
}

CBinaryLabels* CCrossValidatedCalibration::apply_locked_binary(
	SGVector<index_t> subset_indices) {

	index_t num_machines = m_calibration_machines->get_num_elements();

	CBinaryLabels* temp_result;
	CMachine* temp_machine;
	SGVector<float64_t> result_values=NULL;
	CBinaryLabels* result_labels;

	for (index_t i = 0; i < num_machines; ++i) 
	{
		temp_machine = (CMachine*)m_calibration_machines->get_element(i);
		temp_result = (CBinaryLabels*)temp_machine->apply_locked(subset_indices);
		if (result_values == NULL) 
		{
			result_values = temp_result->get_values();
			result_labels = temp_result;
		} 
		else 
		{
			result_values += temp_result->get_values();
		}	
	}

	#pragma omp parallel for
	for (index_t i=0; i< result_values.vlen; ++i)
	{
		result_values[i] = result_values[i]/num_machines;
	}

	result_labels->set_values(result_values);

	return result_labels;
}

CMulticlassLabels* CCrossValidatedCalibration::apply_multiclass(CFeatures* features)
{
	index_t num_machines = m_calibration_machines->get_num_elements();

	index_t num_classes = ((CMulticlassLabels*)m_labels)->get_num_classes();

	CMulticlassLabels* temp_result;
	CMachine* temp_machine;
	SGVector<float64_t> result_values, temp_values;
	CMulticlassLabels* result_labels;


	for (index_t i = 0; i < num_machines; ++i) 
	{
		temp_machine = (CMachine*)m_calibration_machines->get_element(i);
		temp_result = (CMulticlassLabels*)temp_machine->apply(features);
		if (i==0) 
		{
			result_labels = temp_result;
		} 
		else 
		{
			for (index_t j=0; j < num_classes; ++j) 
			{
				result_values = temp_result->get_multiclass_confidences(j);
				temp_values = result_labels->get_multiclass_confidences(j);
				temp_values += result_values;
				result_labels->set_multiclass_confidences(j, temp_values);
			}
		}
	}

	
	for (index_t i=0; i< num_classes; ++i)
	{
		temp_values = result_labels->get_multiclass_confidences(i);

		#pragma omp parallel for
		for (index_t j=0; j<temp_values.vlen; j++) 
		{
			temp_values[j] = temp_values[j]/num_machines;

		}
		result_labels->set_multiclass_confidences(i, temp_values);
		
	}

	return result_labels;

}

CMulticlassLabels* CCrossValidatedCalibration::apply_locked_multiclass(
	SGVector<index_t> subset_indices)
{
	index_t num_machines = m_calibration_machines->get_num_elements();

	index_t num_classes = ((CMulticlassLabels*)m_labels)->get_num_classes();

	CMulticlassLabels* temp_result;
	CMachine* temp_machine;
	SGVector<float64_t> result_values, temp_values;
	CMulticlassLabels* result_labels;


	for (index_t i = 0; i < num_machines; ++i) 
	{
		temp_machine = (CMachine*)m_calibration_machines->get_element(i);
		temp_result = (CMulticlassLabels*)temp_machine->apply_locked_multiclass(subset_indices);
		if (i==0) 
		{
			result_labels = temp_result;
		} 
		else 
		{
			for (index_t j=0; j < num_classes; ++j) 
			{
				result_values = temp_result->get_multiclass_confidences(j);
				temp_values = result_labels->get_multiclass_confidences(j);
				temp_values += result_values;
				result_labels->set_multiclass_confidences(j, temp_values);
			}
		}
	}

	
	for (index_t i=0; i< num_classes; ++i)
	{
		temp_values = result_labels->get_multiclass_confidences(i);

		#pragma omp parallel for
		for (index_t j=0; j<temp_values.vlen; j++) 
		{
			temp_values[j] = temp_values[j]/num_machines;

		}
		result_labels->set_multiclass_confidences(i, temp_values);
		
	}

	return result_labels;

}

CCrossValidatedCalibration::CCrossValidatedCalibration(): CMachine() 
{
	init();
}

CCrossValidatedCalibration::CCrossValidatedCalibration(
	    CMachine* machine, CLabels* labels,
	    CSplittingStrategy* splitting_strategy, 
	    CCalibrationMethod* calibration_method): CMachine() 
{
	init();

	m_machine = machine;
	m_labels = labels;
	m_splitting_strategy = splitting_strategy;
	m_calibration_method = calibration_method;
	m_calibration_method = calibration_method;
}

void CCrossValidatedCalibration::init() {
	m_machine = NULL;
	m_labels = NULL;
	m_splitting_strategy = NULL;
	m_calibration_method = NULL;
}

CCrossValidatedCalibration::~CCrossValidatedCalibration() {
	SG_UNREF(m_machine);
	SG_UNREF(m_labels);
	SG_UNREF(m_splitting_strategy);
	SG_UNREF(m_calibration_method);
}