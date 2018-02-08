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

#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

CCalibrationMethod::CCalibrationMethod()
{
}

CCalibrationMethod::~CCalibrationMethod()
{
}

bool CCalibrationMethod::fit_binary(CBinaryLabels* predictions, CBinaryLabels* targets)
{
	SG_NOTIMPLEMENTED

	return true;
}

CBinaryLabels* CCalibrationMethod::calibrate_binary(CBinaryLabels* predictions)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CCalibrationMethod::fit_multiclass(CMulticlassLabels* predictions, CMulticlassLabels* targets)
{
	SG_NOTIMPLEMENTED
	
	return true;
}

CMulticlassLabels* CCalibrationMethod::calibrate_multiclass(CMulticlassLabels* predictions)
{
	SG_NOTIMPLEMENTED

	return NULL;
}