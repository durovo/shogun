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

#include <shogun/evaluation/Calibration.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/lib/config.h>
#include <shogun/machine/Machine.h>

#ifndef _CROSS_VALIDATED_CALIBRATION_H__
#define _CROSS_VALIDATED_CALIBRATION_H__

namespace shogun
{

	class CCrossValidatedCalibration : public CMachine
	{

	public:
		/** constructor */
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
		    CSplittingStrategy* splitting_strategy,
		    CCalibrationMethod* calibration_method);

		/** destructor */
		virtual ~CCrossValidatedCalibration();

		/** get name
		 *
		 * @return CrossValidatedCalibration
		 */
		virtual const char* get_name() const
		{
			return "CrossValidatedCalibration";
		}

		/** returns problem type of the machine to be calibrated
		 * @return problem type
		 */
		virtual EProblemType get_machine_problem_type() const;

		/** do cross validated calibration on machine
		* @param data on which the machine is to trained
		* @return whether training was successful
		*/
		virtual bool train(CFeatures* data = NULL);

		/** get calibrated result for binary machine
		* @param features the features on which the machine must be applied
		* @return binary labels
		*/
		virtual CBinaryLabels* apply_binary(CFeatures* features = NULL);

		/** get calibrated result for locked binary machine
		* @param subset_indices, indices on which the machine is to be applied
		* @return binary labels
		*/
		virtual CBinaryLabels*
		apply_locked_binary(SGVector<index_t> subset_indices);

		/** do cross validated calibration on locked machine
		* @param indices on which the machine is to trained
		* @return whether training was successful
		*/
		virtual bool train_locked(SGVector<index_t> indices);

		/** get calibrated result for locked multiclass machine
		* @param subset_indices, indices on which the machine is to be applied
		* @return multiclass labels
		*/
		virtual CMulticlassLabels*
		apply_locked_multiclass(SGVector<index_t> subset_indices);

		/** get calibrated result for multiclass machine
		* @param features the features on which the machine must be applied
		* @return multiclass labels
		*/
		virtual CMulticlassLabels* apply_multiclass(CFeatures* features);

		/** get learning machine
		* @return learning machine
		*/
		CMachine* get_machine() const;

	private:
		/** initialize and register variables
		*/
		void init();

		/** helper function to get calibrated multiclass result
		* @param training_data on which to apply machine
		*/
		template <typename T>
		CMulticlassLabels* get_multiclass_result(T training_data);

		/** helper function to get calibrated binary result
		* @param training_data on which to apply machine
		*/
		template <typename T>
		CBinaryLabels* get_binary_result(T training_data);

		/** helper function to get predictions given features
		* @param machine, trained machine
		* @param features, features on which the machine is to be applied
		* @return predicted labels
		*/
		CLabels* apply_once(CMachine* machine, CFeatures* features);

		/** helper function to get predictions given subset indices
		* @param machine, locked trained machine
		* @param subset_indices, indices on which the machine is to be applied
		* @return predicted labels
		*/
		CLabels*
		apply_once(CMachine* machine, SGVector<index_t> subset_indices);

	private:
		/** array of calibration machines*/
		CDynamicObjectArray* m_calibration_machines;
		/** learning machine to be calibrated*/
		CMachine* m_machine;
		/** true labels */
		CLabels* m_labels;
		/** cross validation splitting strategy */
		CSplittingStrategy* m_splitting_strategy;
		/** method to be used for calibration */
		CCalibrationMethod* m_calibration_method;
	};
}
#endif