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

#ifndef _SIGMOID_CALIBRATION_METHOD_H__
#define _SIGMOID_CALIBRATION_METHOD_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>


namespace shogun
{
	/** @brief Calibrates labels based on Platt Scaling [1]. Note that first calibration parameters need to be fitted by calling fit. 
	* Usually this done using the training data and labels. 
	* [1] Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
	* Advances in large margin classifiers. 1999
	*/
	class CSigmoidCalibrationMethod : public CCalibrationMethod
	{
	public:
		/** Constructor. */
		CSigmoidCalibrationMethod();

		/** Destructor. */
		virtual ~CSigmoidCalibrationMethod();

		/** Get name. */
		virtual const char* get_name() const
		{
			return "SigmoidCalibrationMethod";
		}

		/** Fit sigmoid parameters for binary labels.
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

		/** Fit calibration parameters for multiclass labels. Fits sigmoid 
		* parameters for each class seperately.
		* @param predictions The predictions outputted by the machine
		* @param targets The true labels corresponding to the predictions
		* @return boolean indicating whether the calibration was succesful
		**/
		virtual bool fit_multiclass(CMulticlassLabels* predictions, CMulticlassLabels* targets);

		/** Calibrate multiclass predictions based on parameters learned by calling fit.
		* The predictions are normalized over all classes.
		* @param predictions The predictions outputted by the machine
		* @return Calibrated binary labels
		**/
		virtual CMulticlassLabels* calibrate_multiclass(CMulticlassLabels* predictions);

	private:
		/** Initialize parameters */
		void init();

		/** Helper function that calibrates values of given vector using the given sigmoid parameters
		* @param values The values to be calibrated
		* @param params The sigmoid paramters to be used for calibration
		*/
		SGVector<float64_t> calibrate_values(SGVector<float64_t> values, CStatistics::SigmoidParamters params);

	private:
		/** Array to store sigmoid parameters for each class. In case of binary labels, only one pair of parameters are stored. */
		CStatistics::SigmoidParamters* m_sigmoid_parameters;
	};
}
#endif