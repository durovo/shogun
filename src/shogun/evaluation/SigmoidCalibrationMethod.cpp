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

using namespace shogun;

CSigmoidCalibrationMethod::CSigmoidCalibrationMethod() : CCalibrationMethod()
{
	init();
}

CSigmoidCalibrationMethod::~CSigmoidCalibrationMethod()
{
	SG_UNREF(m_sigmoid_parameters);
}

void CSigmoidCalibrationMethod::init()
{
	m_sigmoid_parameters = new CDynamicObjectArray();
	SG_ADD(
	    (CSGObject**)&m_sigmoid_parameters, "m_sigmoid_parameters",
	    "array of sigmoid calibration parameters", MS_NOT_AVAILABLE);
}

bool CSigmoidCalibrationMethod::fit_binary(CBinaryLabels* predictions, CBinaryLabels* targets)
{
	auto params = CStatistics::fit_sigmoid(predictions->get_values(), targets->get_labels());

	SG_UNREF(m_sigmoid_parameters)

	m_sigmoid_parameters = new CDynamicObjectArray(1);

	m_sigmoid_parameters->set_element(params, 0);

	return true;
}

CBinaryLabels*
CSigmoidCalibrationMethod::calibrate_binary(CBinaryLabels* predictions)
{
	auto params = (CStatististics::SigmoidParamters)m_sigmoid_parameters->get_element(0);
	// Convert predictions to probabilties 
	auto values = predictions->get_values();
	for (index_t i = 0; i < values.vlen; ++i)
	{
		float64_t fApB = values[i] * params.a + params.b;
		values[i] = fApB >= 0 ? CMath::exp(-fApB) / (1.0 + CMath::exp(-fApB))
		                      : 1.0 / (1 + CMath::exp(fApB));
	}

	CBinaryLabels* calibrated_predictions = new CBinaryLabels(values);

	return calibrated_predictions;
}