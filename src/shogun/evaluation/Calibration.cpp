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
#include <shogun/evaluation/Calibration.h>
#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/lib/config.h>

using namespace shogun;

CBinaryLabels* CCalibration::apply_binary(CFeatures* features) 
{
	//I don't think that this is necessary
	if (features == NULL) 
	{
		features = m_features;
	}
	CBinaryLabels* result_labels = (CBinaryLabels*)m_machine->apply(features);
	CCalibrationMethod* method = (CCalibrationMethod*)m_calibration_machines->get_element(0);
	SGVector<float64_t> confidence_values = method->apply_binary(result_labels->get_values());
	result_labels->set_values(confidence_values);
	


	return result_labels;
	//return new CBinaryLabels;
	
}

CMulticlassLabels* CCalibration::apply_multiclass(CFeatures* features)
{
	//I don't think that this is necessary
	if (features == NULL) 
	{
		features = m_features;
	}

	index_t num_calibration_machines = m_calibration_machines->get_num_elements();
	CMulticlassLabels* result_labels = (CMulticlassLabels*)m_machine->apply(features);
	for (index_t i=0; i<num_calibration_machines; ++i)
	{
		CCalibrationMethod* method = (CCalibrationMethod*)m_calibration_machines->get_element(i);
		SGVector<float64_t> confidence_values = method->apply_binary(result_labels->get_multiclass_confidences(i));
		result_labels->set_multiclass_confidences(i, confidence_values);
	}
	


	return result_labels;
}

CBinaryLabels* CCalibration::apply_locked_binary(SGVector<index_t> subset_indices) 
{

	CBinaryLabels* result_labels = (CBinaryLabels*)m_machine->apply_locked(subset_indices);
	CCalibrationMethod* method = (CCalibrationMethod*)m_calibration_machines->get_element(0);
	SGVector<float64_t> confidence_values = method->apply_binary(result_labels->get_values());
	result_labels->set_values(confidence_values);

	return result_labels;
}

EProblemType CCalibration::get_machine_problem_type() const
{
	return m_machine->get_machine_problem_type();
}

bool CCalibration::train(CFeatures* features) 
{
	CCalibrationMethod* calibration_machine = NULL;
	if (get_machine_problem_type() == PT_MULTICLASS) 
	{
		SGVector<float64_t> confidences;
		index_t num_calibration_machines = ((CMulticlassLabels*)get_labels())->get_num_classes();
		m_calibration_machines = new CDynamicObjectArray(num_calibration_machines);
		m_machine->train(features);
		CMulticlassLabels* result_labels = (CMulticlassLabels*)m_machine->apply(features);
		for (index_t i=0; i<num_calibration_machines; ++i)
		{
			confidences = result_labels->get_multiclass_confidences(i);

			calibration_machine = (CCalibrationMethod*)m_method->clone();
			if (!calibration_machine->train(confidences))
			{
				return false;
			}
			m_calibration_machines->set_element(calibration_machine, i);

		}
	}
	else 
	{
		SGVector<float64_t> confidences;
		m_calibration_machines = new CDynamicObjectArray(1);
		m_machine->train(features);
		CBinaryLabels* result_labels = (CBinaryLabels*)m_machine->apply_binary(features);

		confidences = result_labels->get_values();

		calibration_machine = (CCalibrationMethod*)m_method->clone();
		if (!calibration_machine->train(confidences))
		{
			return false;
		}
		m_calibration_machines->set_element(calibration_machine, 0);
	}

	return true;
}

bool CCalibration::train_locked(SGVector<index_t> subset_indices) 
{
	CBinaryLabels* m_result_labels = (CBinaryLabels*)m_machine->apply_locked(subset_indices);
	CStatistics::SigmoidParamters params =
		        CStatistics::fit_sigmoid(m_result_labels->get_values());
	a = params.a;
	b = params.b;

	return true;
}

CCalibration::CCalibration(): CMachine()
{
	init();
}

void CCalibration::set_calibration_method(CCalibrationMethod* method) 
{
	m_method = method;
}

void CCalibration::set_machine(CMachine* machine)
{
	m_machine = machine;
}

void CCalibration::init() 
{
	m_machine = NULL;
	m_labels = NULL;
}

CCalibration::~CCalibration() {
	SG_UNREF(m_machine);
	SG_UNREF(m_labels);
}

CMachine* CCalibration::get_machine()
{
	return m_machine;
}