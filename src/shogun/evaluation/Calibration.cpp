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
#include <shogun/lib/config.h>

using namespace shogun;

CBinaryLabels* CCalibration::apply_binary(CFeatures* features) 
{
	//I don't think that this is necessary
	if (features == NULL) 
	{
		features = m_features;
	}
	//Code taken from CBinaryLabels
	CBinaryLabels* m_result = (CBinaryLabels*)m_machine->apply_binary(features);
	SGVector<float64_t> m_result_labels = m_result->get_values();
	for (index_t i = 0; i < m_result_labels.vlen; ++i)
	{
		float64_t fApB = m_result_labels[i] * a + b;
		m_result_labels[i] = fApB >= 0 ? CMath::exp(-fApB) / (1.0 + CMath::exp(-fApB)) :
		                      1.0 / (1 + CMath::exp(fApB));
	}
	m_result->set_values(m_result_labels);

	return m_result;
	//return new CBinaryLabels;
	
}

CBinaryLabels* CCalibration::apply_locked_binary(SGVector<index_t> subset_indices) 
{
	CBinaryLabels* m_result = (CBinaryLabels*)m_machine->apply_locked(subset_indices);
	SGVector<float64_t> m_result_labels = m_result->get_values();

	for (index_t i = 0; i < m_result_labels.vlen; ++i)
	{
		float64_t fApB = m_result_labels[i] * a + b;
		m_result_labels[i] = fApB >= 0 ? CMath::exp(-fApB) / (1.0 + CMath::exp(-fApB)) :
		                      1.0 / (1 + CMath::exp(fApB));
	}
	
	m_result->set_values(m_result_labels);

	return m_result;
}

EProblemType CCalibration::get_machine_problem_type() const
{
	return m_machine->get_machine_problem_type();
}

bool CCalibration::train(CFeatures* features) 
{
	CBinaryLabels* m_result_labels = (CBinaryLabels*)m_machine->apply(features);
	CStatistics::SigmoidParamters params =
		        CStatistics::fit_sigmoid(m_result_labels->get_values());
	a = params.a;
	b = params.b;

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