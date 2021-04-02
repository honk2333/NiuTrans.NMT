#include "RoughTopK.h"
#include "../Utility.h"
#include "../../niutensor/tensor/core/CHeader.h"

using namespace nts;

template<typename T, int beam_size>
__global__
void select_beam_rough_topk(T *prob, T *seq_probs, T *seq_score, T *scoreTopK, T *index) {
    if (cur_step != 0 && alive_seq[blockIdx.x * max_step + cur_step] == end_id) {
        // this is a finished beam
        if (threadIdx.x == 0) {
            num_beam_can[blockIdx.x + 1] = 1;      // generate one candidate
            int pos = atomicAdd(num_beam_can, 1);  // get a candidate pos
            if (diverse_lambda == 0) {
                can_score[pos] =
                        seq_score[blockIdx.x];  // this beam's score will not be change
            } else {
                // add the beam id offset in score to sort in each beam
                int batch_id = blockIdx.x / beam_size;
                can_score[pos] = seq_score[blockIdx.x] +
                                 (blockIdx.x - batch_id) * min_log_probability;
            }
            can_idx[pos] = end_id + (blockIdx.x % beam_size) * vocab_size;  // EOS
        }
        return;
    }

    /* step1: compute each thread's max_logit and sum_exp_logit, store in
     * rough_top_kth_logit, sum_exp_logit */
    const int block_start = blockIdx.x * vocab_size;
    const int left_idx = block_start + threadIdx.x;
    const int right_idx = (blockIdx.x + 1) * vocab_size;
    float rough_top_kth_logit = CUDA_FLOAT_INF_NEG;
    float sum_exp_logit = 0;
    for (int i = left_idx; i < right_idx; i += blockDim.x) {
        float lgt = (float) logits[i] + (float) __ldg(&logit_bias[i - block_start]);
        rough_top_kth_logit = fmaxf(rough_top_kth_logit, lgt);
    }
    float max_logit = blockReduceMax(rough_top_kth_logit);
    __shared__ float s_max_logit;
    if (threadIdx.x == 0) {
        s_max_logit = max_logit;
    }
    __syncthreads();
    for (int i = left_idx; i < right_idx; i += blockDim.x) {
        float lgt =
                fmaxf((float) (logits[i]) + (float) __ldg(&logit_bias[i - block_start]) -
                      s_max_logit,
                      logit_thresh_min);
        sum_exp_logit += expf(lgt);
    }

    /*
    step2: compute rough top-kth-logits and sum_exp_logit among the whole beam,
    saved into s_topk and
        s_log_prob_base
    */
    __shared__ float
            s_log_prob_base;      // prefix sequence log prob - log_sum_exp_logit
    __shared__ float s_topk;  // rough top k-th value of logits
    __shared__ int num_cur_beam_can;  // candidate number for this beam
    sum_exp_logit = blockReduceSum(sum_exp_logit);
    rough_top_kth_logit = blockRoughTopK<float, beam_size>(rough_top_kth_logit);
    if (threadIdx.x == 0) {
        s_log_prob_base = seq_probs[blockIdx.x] - logf(sum_exp_logit) - s_max_logit;
        s_topk = rough_top_kth_logit;
        num_cur_beam_can = 0;
    }

    /*
    step3 : select the candidate token with logits bigger than s_topk,
            compute the seq probability ended with them,
        save the probability, token_index, selected token number.
    */
    int idx = left_idx;
    int batch_id = blockIdx.x / beam_size;
    int batch_start_pos = batch_id * beam_size * vocab_size;
    // int unk_vocab_id = vocab_size - 3;  // last three element: unk, start, eos
    __shared__ int l_n;  // current iteration candidate number
    for (int iter = 0; iter < (vocab_size + blockDim.x - 1) / blockDim.x;
         iter++) {
        // zero the counter
        if (threadIdx.x == 0) l_n = 0;
        __syncthreads();

        float lgt = CUDA_FLOAT_INF_NEG - 1.f;  // min s_topk is CUDA_FLOAT_INF_NEG
        int pos;
        int vocab_id = idx - block_start;

        // if ((vocab_id < vocab_size) && (vocab_id != unk_vocab_id)) {
        if (vocab_id < vocab_size) {
            lgt = (float) (logits[idx]) + (float) __ldg(&logit_bias[vocab_id]);
            if (lgt >= s_topk)
                // pos: relative pos inside this iteration
                pos = atomicAdd(&l_n, 1);
        }
        __syncthreads();

        // leader increments the global counter
        if (threadIdx.x == 0) {
            atomicAdd(&num_cur_beam_can, l_n);
            l_n = atomicAdd(num_beam_can, l_n);
        }
        __syncthreads();

        // threads with true predicates write their elements
        if ((lgt >= s_topk)) {
            pos += l_n;  // increment local pos by global counter
            if (diverse_lambda == 0) {
                can_score[pos] = fmaxf((lgt + s_log_prob_base) * length_norm,
                                       min_log_probability + 1.f) +
                                 batch_id * min_log_probability;
            } else {
                can_score[pos] = fmaxf((lgt + s_log_prob_base) * length_norm,
                                       min_log_probability + 1.f) +
                                 blockIdx.x * min_log_probability;
            }
            can_idx[pos] = idx - batch_start_pos;
        }
        __syncthreads();
        idx += blockDim.x;
    }
    if (threadIdx.x == 0) {
        num_beam_can[blockIdx.x + 1] = num_cur_beam_can;
    }
}

template<typename T>
void select_beam_rough_topk_launcher(
        const XTensor prob, const XTensor seq_probs, const XTensor seq_score, XTensor scoreTopK, XTensor index,
        int beam_size) {
    step_token_num = prob.dimSize[prob.order - 1];
    max_thread_per_block = 512;
    select_beam_rough_topk<T, beam_size>
    <<<step_token_num, max_thread_per_block>>>((DTYPE *) prob.data, (DTYPE *) seq_probs.data,
                                               (DTYPE *) seq_score.data,
                                               (DTYPE *) scoreTopK.data, (int *) index.data,);
}