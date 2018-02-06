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
#include <gtest/gtest.h>

using namespace shogun;

TEST(SigmoidCalibrationMethodTest, binary_calibration)
{
	CMath::init_random(8);
	SGVector<float64_t> preds(10), labs(10);

	preds.vector[0] = 0.6;
	preds.vector[1] = -0.2;
	preds.vector[2] = 0.7;
	preds.vector[3] = 0.9;
	preds.vector[4] = -0.1;
	preds.vector[5] = -0.3;
	preds.vector[6] = 0.9;
	preds.vector[7] = 0.6;
	preds.vector[8] = -0.3;
	preds.vector[9] = 0.7;

	labs.vector[0] = 1;
	labs.vector[1] = -1;
	labs.vector[2] = 1;
	labs.vector[3] = 1;
	labs.vector[4] = -1;
	labs.vector[5] = -1;
	labs.vector[6] = 1;
	labs.vector[7] = 1;
	labs.vector[8] = -1;
	labs.vector[9] = -1;

	CBinaryLabels* predictions = new CBinaryLabels(preds);
	CBinaryLabels* labels = new CBinaryLabels(labs);

	SG_REF(predictions)

	CSigmoidCalibrationMethod* sigmoid_calibration = new CSigmoidCalibrationMethod();

	auto calibrated = sigmoid_calibration->fit_binary(predictions, labels);

	EXPECT_EQ(calibrated, true);

	auto calibrated_labels = sigmoid_calibration->calibrate_binary(predictions);

	auto values = calibrated_labels->get_values();

	EXPECT_EQ(values[0], 0.656628663983293337);
	EXPECT_EQ(values[1], 0.159375349583615822);
	EXPECT_EQ(values[2], 0.718534704684106407);
	EXPECT_EQ(values[3], 0.819801347075004516);
	EXPECT_EQ(values[4], 0.201976857736835741);
	EXPECT_EQ(values[5], 0.124359200656053326);
	EXPECT_EQ(values[6], 0.819801347075004516);
	EXPECT_EQ(values[7], 0.656628663983293337);
	EXPECT_EQ(values[8], 0.124359200656053326);
	EXPECT_EQ(values[9], 0.718534704684106407);

	SG_UNREF(sigmoid_calibration)
	SG_UNREF(predictions)
	SG_UNREF(labels)
	SG_UNREF(calibrated_labels)
}