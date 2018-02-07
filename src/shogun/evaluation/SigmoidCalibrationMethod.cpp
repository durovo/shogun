/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2018 Dhruv Arya
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those
 * of the authors and should not be interpreted as representing official
 * policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/evaluation/SigmoidCalibrationMethod.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CSigmoidCalibrationMethod::CSigmoidCalibrationMethod() : CCalibrationMethod()
{
	init();
}

CSigmoidCalibrationMethod::~CSigmoidCalibrationMethod()
{
	delete(m_sigmoid_parameters);
}

void CSigmoidCalibrationMethod::init()
{
	// m_sigmoid_parameters = new CDynamicObjectArray();
	// SG_ADD(
	//     (CSGObject**)&m_sigmoid_parameters, "m_sigmoid_parameters",
	//     "array of sigmoid calibration parameters", MS_NOT_AVAILABLE);
	m_sigmoid_parameters = SG_ALLOC(CStatistics::SigmoidParamters, 1);
	// SG_ADD(m_sigmoid_parameters, "m_sigmoid_parameters",
	//     "array of sigmoid calibration parameters", MS_NOT_AVAILABLE);
}

bool CSigmoidCalibrationMethod::fit_binary(CBinaryLabels* predictions, CBinaryLabels* targets)
{
	auto params = CStatistics::fit_sigmoid(predictions->get_values(), targets->get_labels());

	SG_FREE(m_sigmoid_parameters)

	m_sigmoid_parameters = new CStatistics::SigmoidParamters[1];

	m_sigmoid_parameters[0] = params;

	return true;
}

CBinaryLabels*
CSigmoidCalibrationMethod::calibrate_binary(CBinaryLabels* predictions)
{
	auto params = m_sigmoid_parameters[0];
	// Convert predictions to probabilties 
	auto values = calibrate_values(predictions->get_values(), params);
	
	CBinaryLabels* calibrated_predictions = new CBinaryLabels(values);

	return calibrated_predictions;
}

bool CSigmoidCalibrationMethod::fit_multiclass(CMulticlassLabels* predictions, CMulticlassLabels* targets)
{
	index_t num_classes =
	    predictions->get_num_classes();
	SG_FREE(m_sigmoid_parameters)
	m_sigmoid_parameters =
	    SG_ALLOC(CStatistics::SigmoidParamters, num_classes);

	SGVector<float64_t> confidences;

	for (index_t i = 0; i < num_classes; ++i)
	{
		auto class_predictions = predictions->get_binary_for_class(i);
		auto class_labels = targets->get_binary_for_class(i);
		auto pred_values = class_predictions->get_values();
		auto target_labels = class_labels->get_labels();
		m_sigmoid_parameters[i] = CStatistics::fit_sigmoid(pred_values, target_labels);
	}

	return true;
}

CMulticlassLabels*
CSigmoidCalibrationMethod::calibrate_multiclass(CMulticlassLabels* predictions)
{
	index_t num_classes = predictions->get_num_classes();

	auto result_labels = (CMulticlassLabels*) predictions->clone();

	for (index_t i = 0; i < num_classes; ++i)
	{
		auto binary_predictions = predictions->get_binary_for_class(i);
		auto class_values = binary_values->get_values();
		SGVector<float64_t> calibrated_values =
		    calibrate_values(class_values, m_sigmoid_parameters[i]);
		result_labels->set_multiclass_confidences(i, calibrated_values);
	}

	SGVector<float64_t> temp_confidences =
	    result_labels->get_multiclass_confidences(0);
	temp_confidences.zero();

	index_t num_samples = temp_confidences.vlen;

// normalize the probabilities
#pragma omp parallel for
	for (index_t i = 0; i < num_classes; ++i)
	{
		SGVector<float64_t> confidence_values =
		    result_labels->get_multiclass_confidences(i);
		float64_t sum = SGVector<float64_t>::sum(confidence_values);
		sum += 1E-10;
		linalg::scale(confidence_values, confidence_values, 1/sum);
		result_labels->set_multiclass_confidences(confidence_values, i);
	}
	return result_labels;
}

SGVector<float64_t>
CSigmoidCalibrationMethod::calibrate_values(SGVector<float64_t> values, CStatistics::SigmoidParamters params)
{
	for (index_t i = 0; i < values.vlen; ++i)
	{
		float64_t fApB = values[i] * params.a + params.b;
		values[i] = fApB >= 0 ? CMath::exp(-fApB) / (1.0 + CMath::exp(-fApB))
		                      : 1.0 / (1 + CMath::exp(fApB));
	}
	return values;
}