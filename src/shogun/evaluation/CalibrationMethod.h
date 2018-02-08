/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
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

#ifndef _CALIBRATION_METHOD_H__
#define _CALIBRATION_METHOD_H__

#include <shogun/lib/config.h>

#include <shogun/machine/Machine.h>

namespace shogun
{
	/** @brief Base class for all calibration methods. Call fit to 
	fit the parameters on the predictions and the true labels. Call calibrate to calibrate predictions. **/
	class CCalibrationMethod : public CSGObject
	{
	public:
		/** Constructor. */
		CCalibrationMethod();
		/** Destructor. */
		virtual ~CCalibrationMethod();

		virtual const char* get_name() const
		{
			return "CalibrationMethod";
		}

		/** Fit calibration parameters for binary labels.
		* @param predictions The predictions outputted by the machine
		* @param targets The true labels corresponding to the predictions
		* @return boolean indicating whether the calibration was succesful
		**/
		virtual bool fit_binary(CBinaryLabels* predictions, CBinaryLabels* targets);

		/** Calibrate binary predictions based on parameters learned by calling fit.
		* @param predictions The predictions outputted by the machine
		* @return Calibrated binary labels
		**/
		virtual CBinaryLabels* calibrate_binary(CBinaryLabels* predictions);

		/** Fit calibration parameters for multiclass labels.
		* @param predictions The predictions outputted by the machine
		* @param targets The true labels corresponding to the predictions
		* @return boolean indicating whether the calibration was succesful
		**/
		virtual bool fit_multiclass(CMulticlassLabels* predictions, CMulticlassLabels* targets);

		/** Calibrate multiclass predictions based on parameters learned by calling fit.
		* @param predictions The predictions outputted by the machine
		* @return Calibrated binary labels
		**/
		virtual CMulticlassLabels* calibrate_multiclass(CMulticlassLabels* predictions);
	};
}
#endif