/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

using namespace shogun;

CWeightedCommWordStringKernel::CWeightedCommWordStringKernel(
	int32_t size, bool us)
: CCommWordStringKernel(size, us), degree(0), weights(NULL)
{
	init_dictionary(1<<(sizeof(uint16_t)*9));
	ASSERT(us==false);
}

CWeightedCommWordStringKernel::CWeightedCommWordStringKernel(
	CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r, bool us,
	int32_t size)
: CCommWordStringKernel(size, us), degree(0), weights(NULL)
{
	init_dictionary(1<<(sizeof(uint16_t)*9));
	ASSERT(us==false);

	init(l,r);
}

CWeightedCommWordStringKernel::~CWeightedCommWordStringKernel()
{
	delete[] weights;
}

bool CWeightedCommWordStringKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(((CStringFeatures<uint16_t>*) l)->get_order() ==
			((CStringFeatures<uint16_t>*) r)->get_order());
	degree=((CStringFeatures<uint16_t>*) l)->get_order();
	set_wd_weights();

	CCommWordStringKernel::init(l,r);
	return init_normalizer();
}

void CWeightedCommWordStringKernel::cleanup()
{
	delete[] weights;
	weights=NULL;

	CCommWordStringKernel::cleanup();
}

bool CWeightedCommWordStringKernel::set_wd_weights()
{
	delete[] weights;
	weights=new float64_t[degree];

	int32_t i;
	float64_t sum=0;
	for (i=0; i<degree; i++)
	{
		weights[i]=degree-i;
		sum+=weights[i];
	}
	for (i=0; i<degree; i++)
		weights[i]=CMath::sqrt(weights[i]/sum);

	return weights!=NULL;
}

bool CWeightedCommWordStringKernel::set_weights(float64_t* w, int32_t d)
{
	ASSERT(d==degree);

	delete[] weights;
	weights=new float64_t[degree];
	for (int32_t i=0; i<degree; i++)
		weights[i]=CMath::sqrt(w[i]);
	return true;
}
  
float64_t CWeightedCommWordStringKernel::compute_helper(
	int32_t idx_a, int32_t idx_b, bool do_sort)
{
	int32_t alen, blen;

	CStringFeatures<uint16_t>* l = (CStringFeatures<uint16_t>*) lhs;
	CStringFeatures<uint16_t>* r = (CStringFeatures<uint16_t>*) rhs;

	uint16_t* av=l->get_feature_vector(idx_a, alen);
	uint16_t* bv=r->get_feature_vector(idx_b, blen);

	uint16_t* avec=av;
	uint16_t* bvec=bv;

	if (do_sort)
	{
		if (alen>0)
		{
			avec=new uint16_t[alen];
			memcpy(avec, av, sizeof(uint16_t)*alen);
			CMath::radix_sort(avec, alen);
		}
		else
			avec=NULL;

		if (blen>0)
		{
			bvec=new uint16_t[blen];
			memcpy(bvec, bv, sizeof(uint16_t)*blen);
			CMath::radix_sort(bvec, blen);
		}
		else
			bvec=NULL;
	}
	else
	{
		if ( (l->get_num_preproc() != l->get_num_preprocessed()) ||
				(r->get_num_preproc() != r->get_num_preprocessed()))
		{
			SG_ERROR("not all preprocessors have been applied to training (%d/%d)"
					" or test (%d/%d) data\n", l->get_num_preprocessed(), l->get_num_preproc(),
					r->get_num_preprocessed(), r->get_num_preproc());
		}
	}

	float64_t result=0;
	uint8_t mask=0;

	for (int32_t d=0; d<degree; d++)
	{
		mask = mask | (1 << (degree-d-1));
		uint16_t masked=((CStringFeatures<uint16_t>*) lhs)->get_masked_symbols(0xffff, mask);

		int32_t left_idx=0;
		int32_t right_idx=0;
		float64_t weight=weights[d]*weights[d];

		while (left_idx < alen && right_idx < blen)
		{
			uint16_t lsym=avec[left_idx] & masked;
			uint16_t rsym=bvec[right_idx] & masked;

			if (lsym == rsym)
			{
				int32_t old_left_idx=left_idx;
				int32_t old_right_idx=right_idx;

				while (left_idx<alen && (avec[left_idx] & masked) ==lsym)
					left_idx++;

				while (right_idx<blen && (bvec[right_idx] & masked) ==lsym)
					right_idx++;

				result+=weight*(left_idx-old_left_idx)*(right_idx-old_right_idx);
			}
			else if (lsym<rsym)
				left_idx++;
			else
				right_idx++;
		}
	}

	if (do_sort)
	{
		delete[] avec;
		delete[] bvec;
	}

	return result;
}

void CWeightedCommWordStringKernel::add_to_normal(
	int32_t vec_idx, float64_t weight)
{
	int32_t len=-1;
	CStringFeatures<uint16_t>* s=(CStringFeatures<uint16_t>*) lhs;
	uint16_t* vec=s->get_feature_vector(vec_idx, len);

	if (len>0)
	{
		for (int32_t j=0; j<len; j++)
		{
			uint8_t mask=0;
			int32_t offs=0;
			for (int32_t d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				int32_t idx=s->get_masked_symbols(vec[j], mask);
				idx=s->shift_symbol(idx, degree-d-1);
				dictionary_weights[offs + idx] += normalizer->normalize_lhs(weight*weights[d], vec_idx);
				offs+=s->shift_offset(1,d+1);
			}
		}

		set_is_initialized(true);
	}
}

void CWeightedCommWordStringKernel::merge_normal()
{
	ASSERT(get_is_initialized());
	ASSERT(use_sign==false);

	CStringFeatures<uint16_t>* s=(CStringFeatures<uint16_t>*) rhs;
	uint32_t num_symbols=(uint32_t) s->get_num_symbols();
	int32_t dic_size=1<<(sizeof(uint16_t)*8);
	float64_t* dic=new float64_t[dic_size];
	memset(dic, 0, sizeof(float64_t)*dic_size);

	for (uint32_t sym=0; sym<num_symbols; sym++)
	{
		float64_t result=0;
		uint8_t mask=0;
		int32_t offs=0;
		for (int32_t d=0; d<degree; d++)
		{
			mask = mask | (1 << (degree-d-1));
			int32_t idx=s->get_masked_symbols(sym, mask);
			idx=s->shift_symbol(idx, degree-d-1);
			result += dictionary_weights[offs + idx];
			offs+=s->shift_offset(1,d+1);
		}
		dic[sym]=result;
	}

	init_dictionary(1<<(sizeof(uint16_t)*8));
	memcpy(dictionary_weights, dic, sizeof(float64_t)*dic_size);
	delete[] dic;
}

float64_t CWeightedCommWordStringKernel::compute_optimized(int32_t i)
{ 
	if (!get_is_initialized())
		SG_ERROR( "CCommWordStringKernel optimization not initialized\n");

	ASSERT(use_sign==false);

	float64_t result=0;
	int32_t len=-1;
	CStringFeatures<uint16_t>* s=(CStringFeatures<uint16_t>*) rhs;
	uint16_t* vec=s->get_feature_vector(i, len);

	if (vec && len>0)
	{
		for (int32_t j=0; j<len; j++)
		{
			uint8_t mask=0;
			int32_t offs=0;
			for (int32_t d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				int32_t idx=s->get_masked_symbols(vec[j], mask);
				idx=s->shift_symbol(idx, degree-d-1);
				result += dictionary_weights[offs + idx]*weights[d];
				offs+=s->shift_offset(1,d+1);
			}
		}

		result=normalizer->normalize_rhs(result, i);
	}
	return result;
}

float64_t* CWeightedCommWordStringKernel::compute_scoring(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym, float64_t* target,
	int32_t num_suppvec, int32_t* IDX, float64_t* alphas, bool do_init)
{
	if (do_init)
		CCommWordStringKernel::init_optimization(num_suppvec, IDX, alphas);

	int32_t dic_size=1<<(sizeof(uint16_t)*9);
	float64_t* dic=new float64_t[dic_size];
	memcpy(dic, dictionary_weights, sizeof(float64_t)*dic_size);

	merge_normal();
	float64_t* result=CCommWordStringKernel::compute_scoring(max_degree, num_feat,
			num_sym, target, num_suppvec, IDX, alphas, false);

	init_dictionary(1<<(sizeof(uint16_t)*9));
	memcpy(dictionary_weights,dic,  sizeof(float64_t)*dic_size);
	delete[] dic;

	return result;
}
