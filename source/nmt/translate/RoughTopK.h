#ifndef __ROUGHTOPK_H__
#define __ROUGHTOPK_H__

#include "../Utility.h"
#include "../../niutensor/tensor/core/CHeader.h"

using namespace std;

template<typename T>
void select_beam_rough_topk_launcher(
        const XTensor prob, const XTensor seq_probs, const XTensor seq_score, XTensor scoreTopK, XTensor index,
        int beam_size, int cur_step);

#endif

