// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// sdpa prefill, several query tokens per head, aarch64 only
//
// key and value of each kv head are converted to fp32 and packed once,
// the query heads that share the kv head reuse them
//   packed key   [key panel of 8][head_dim][8 keys]
//   packed value [value group of 8][key][8 dims]
// both are zero padded to a multiple of 8 keys, value_dim is zero padded to a multiple of 8
//
// each task handles a tile of 12 or 8 query rows of one head, walks the keys block by block
// and merges the blocks with online softmax
// the storage type T and output type TO use the element access helpers of sdpa_decode.h

#if __aarch64__

static int sdpa_prefill_get_optimal_tile_n(int head_dim, int value_dim_padded, int key_seqlen)
{
    // keep a block of packed key, packed value and scores around 64KB
    const int block_size = 64 * 1024;
    const int size_per_key = (head_dim + value_dim_padded + 12) * (int)sizeof(float);
    int TILE_N = block_size / size_per_key;
    TILE_N = std::max(8, std::min(TILE_N, 512)) / 8 * 8;

    if (key_seqlen > 0)
    {
        // spread keys evenly over the blocks
        const int num_blocks = (key_seqlen + TILE_N - 1) / TILE_N;
        TILE_N = (key_seqlen + num_blocks - 1) / num_blocks;
        TILE_N = (TILE_N + 7) / 8 * 8;
    }

    return TILE_N;
}

// pk[key_seqlen_padded / 8][head_dim][8]
template<typename T>
static void sdpa_prefill_pack_key(const Mat& key_head, float* pk, int key_seqlen_padded)
{
    const int head_dim = key_head.w;
    const int key_seqlen = key_head.h;

    for (int j = 0; j < key_seqlen_padded; j += 8)
    {
        float* pp = pk + (size_t)j * head_dim;
        const int nk = std::min(8, key_seqlen - j);

        if (nk == 8)
        {
            const T* k0 = key_head.row<const T>(j);
            const T* k1 = k0 + head_dim;
            const T* k2 = k1 + head_dim;
            const T* k3 = k2 + head_dim;
            const T* k4 = k3 + head_dim;
            const T* k5 = k4 + head_dim;
            const T* k6 = k5 + head_dim;
            const T* k7 = k6 + head_dim;

            int d = 0;
            for (; d + 3 < head_dim; d += 4)
            {
                float32x4_t _k0 = sdpa_load4(k0 + d);
                float32x4_t _k1 = sdpa_load4(k1 + d);
                float32x4_t _k2 = sdpa_load4(k2 + d);
                float32x4_t _k3 = sdpa_load4(k3 + d);
                float32x4_t _k4 = sdpa_load4(k4 + d);
                float32x4_t _k5 = sdpa_load4(k5 + d);
                float32x4_t _k6 = sdpa_load4(k6 + d);
                float32x4_t _k7 = sdpa_load4(k7 + d);
                transpose4x4_ps(_k0, _k1, _k2, _k3);
                transpose4x4_ps(_k4, _k5, _k6, _k7);
                vst1q_f32(pp, _k0);
                vst1q_f32(pp + 4, _k4);
                vst1q_f32(pp + 8, _k1);
                vst1q_f32(pp + 12, _k5);
                vst1q_f32(pp + 16, _k2);
                vst1q_f32(pp + 20, _k6);
                vst1q_f32(pp + 24, _k3);
                vst1q_f32(pp + 28, _k7);
                pp += 32;
            }
            for (; d < head_dim; d++)
            {
                pp[0] = sdpa_cvt(k0[d]);
                pp[1] = sdpa_cvt(k1[d]);
                pp[2] = sdpa_cvt(k2[d]);
                pp[3] = sdpa_cvt(k3[d]);
                pp[4] = sdpa_cvt(k4[d]);
                pp[5] = sdpa_cvt(k5[d]);
                pp[6] = sdpa_cvt(k6[d]);
                pp[7] = sdpa_cvt(k7[d]);
                pp += 8;
            }
        }
        else
        {
            for (int d = 0; d < head_dim; d++)
            {
                for (int t = 0; t < 8; t++)
                {
                    pp[t] = t < nk ? sdpa_cvt(key_head.row<const T>(j + t)[d]) : 0.f;
                }
                pp += 8;
            }
        }
    }
}

// pv[value_dim_padded / 8][key_seqlen_padded][8]
template<typename T>
static void sdpa_prefill_pack_value(const Mat& value_head, float* pv, int key_seqlen_padded, int value_dim_padded)
{
    const int value_dim = value_head.w;
    const int key_seqlen = value_head.h;
    const size_t group_stride = (size_t)key_seqlen_padded * 8;

    for (int j = 0; j < key_seqlen_padded; j++)
    {
        float* pp = pv + (size_t)j * 8;

        if (j >= key_seqlen)
        {
            for (int g = 0; g < value_dim_padded / 8; g++)
            {
                vst1q_f32(pp + g * group_stride, vdupq_n_f32(0.f));
                vst1q_f32(pp + g * group_stride + 4, vdupq_n_f32(0.f));
            }
            continue;
        }

        const T* vptr = value_head.row<const T>(j);

        int d = 0;
        for (; d + 7 < value_dim; d += 8)
        {
            float* p = pp + (d / 8) * group_stride;
            vst1q_f32(p, sdpa_load4(vptr + d));
            vst1q_f32(p + 4, sdpa_load4(vptr + d + 4));
        }
        if (d < value_dim)
        {
            float* p = pp + (d / 8) * group_stride;
            for (int k = 0; k < 8; k++)
            {
                p[k] = d + k < value_dim ? sdpa_cvt(vptr[d + k]) : 0.f;
            }
        }
    }
}

// qT[head_dim][MR] with MR 8 or 12, rows past the end are zero
template<typename T>
static void sdpa_prefill_pack_query(const Mat& query_head, float* qT, int i0, int ni, int MR, float scale)
{
    const int head_dim = query_head.w;

    if (ni == MR)
    {
        float32x4_t _scale = vdupq_n_f32(scale);
        for (int r = 0; r < MR; r += 4)
        {
            const T* q0 = query_head.row<const T>(i0 + r);
            const T* q1 = q0 + head_dim;
            const T* q2 = q1 + head_dim;
            const T* q3 = q2 + head_dim;

            float* pp = qT + r;
            int d = 0;
            for (; d + 3 < head_dim; d += 4)
            {
                float32x4_t _q0 = sdpa_load4(q0 + d);
                float32x4_t _q1 = sdpa_load4(q1 + d);
                float32x4_t _q2 = sdpa_load4(q2 + d);
                float32x4_t _q3 = sdpa_load4(q3 + d);
                transpose4x4_ps(_q0, _q1, _q2, _q3);
                vst1q_f32(pp, vmulq_f32(_q0, _scale));
                vst1q_f32(pp + MR, vmulq_f32(_q1, _scale));
                vst1q_f32(pp + MR * 2, vmulq_f32(_q2, _scale));
                vst1q_f32(pp + MR * 3, vmulq_f32(_q3, _scale));
                pp += MR * 4;
            }
            for (; d < head_dim; d++)
            {
                pp[0] = sdpa_cvt(q0[d]) * scale;
                pp[1] = sdpa_cvt(q1[d]) * scale;
                pp[2] = sdpa_cvt(q2[d]) * scale;
                pp[3] = sdpa_cvt(q3[d]) * scale;
                pp += MR;
            }
        }
    }
    else
    {
        for (int d = 0; d < head_dim; d++)
        {
            for (int i = 0; i < MR; i++)
            {
                qT[d * MR + i] = i < ni ? sdpa_cvt(query_head.row<const T>(i0 + i)[d]) * scale : 0.f;
            }
        }
    }
}

// add mask[row][key] of MR rows x 8 keys to pS[key][MR]
template<typename T>
static void sdpa_prefill_add_mask(float* pS, const T* const* maskptrs, int jn, int ni, int nk, int MR)
{
    if (ni == MR && nk == 8)
    {
        for (int r = 0; r < MR; r += 4)
        {
            float32x4_t _r0a = sdpa_load4(maskptrs[r] + jn);
            float32x4_t _r1a = sdpa_load4(maskptrs[r + 1] + jn);
            float32x4_t _r2a = sdpa_load4(maskptrs[r + 2] + jn);
            float32x4_t _r3a = sdpa_load4(maskptrs[r + 3] + jn);
            float32x4_t _r0b = sdpa_load4(maskptrs[r] + jn + 4);
            float32x4_t _r1b = sdpa_load4(maskptrs[r + 1] + jn + 4);
            float32x4_t _r2b = sdpa_load4(maskptrs[r + 2] + jn + 4);
            float32x4_t _r3b = sdpa_load4(maskptrs[r + 3] + jn + 4);
            transpose4x4_ps(_r0a, _r1a, _r2a, _r3a);
            transpose4x4_ps(_r0b, _r1b, _r2b, _r3b);
            float* p = pS + r;
            vst1q_f32(p, vaddq_f32(vld1q_f32(p), _r0a));
            vst1q_f32(p + MR, vaddq_f32(vld1q_f32(p + MR), _r1a));
            vst1q_f32(p + MR * 2, vaddq_f32(vld1q_f32(p + MR * 2), _r2a));
            vst1q_f32(p + MR * 3, vaddq_f32(vld1q_f32(p + MR * 3), _r3a));
            vst1q_f32(p + MR * 4, vaddq_f32(vld1q_f32(p + MR * 4), _r0b));
            vst1q_f32(p + MR * 5, vaddq_f32(vld1q_f32(p + MR * 5), _r1b));
            vst1q_f32(p + MR * 6, vaddq_f32(vld1q_f32(p + MR * 6), _r2b));
            vst1q_f32(p + MR * 7, vaddq_f32(vld1q_f32(p + MR * 7), _r3b));
        }
    }
    else
    {
        // partial rows or keys, rows and keys past the end are left as they are
        for (int t = 0; t < nk; t++)
        {
            for (int i = 0; i < ni; i++)
            {
                pS[t * MR + i] += sdpa_cvt(maskptrs[i][jn + t]);
            }
        }
    }
}

// 12 query rows of one head against all keys of its kv head
// qT[head_dim][12] is prescaled, pk and pv are the packed key and value of the kv head
// maskptrs[12] are the mask rows of the 12 query rows, or all null
// ni is the number of valid query rows
// workspace: scoreT[block_n][12] outT[value_dim_padded][12]
// the result is outT[value_dim_padded][12] and l[12] (denominators)
template<typename T>
static void sdpa_prefill_attention_tile_12(const float* qT, const float* pk, const float* pv, const T* const* maskptrs, int ni, int head_dim, int value_dim_padded, int key_seqlen, int key_seqlen_padded, int block_n, float* scoreT, float* outT, float* l)
{
    const size_t group_stride = (size_t)key_seqlen_padded * 8;

    float32x4_t _m0 = vdupq_n_f32(-FLT_MAX);
    float32x4_t _m1 = vdupq_n_f32(-FLT_MAX);
    float32x4_t _m2 = vdupq_n_f32(-FLT_MAX);
    float32x4_t _l0 = vdupq_n_f32(0.f);
    float32x4_t _l1 = vdupq_n_f32(0.f);
    float32x4_t _l2 = vdupq_n_f32(0.f);
    memset(outT, 0, value_dim_padded * 12 * sizeof(float));

    for (int n = 0; n < key_seqlen; n += block_n)
    {
        const int max_jj = std::min(key_seqlen - n, block_n);

        // qk, scoreT[j][12 rows]
        for (int j = 0; j < max_jj; j += 8)
        {
            const float* pQ = qT;
            const float* pK = pk + (size_t)(n + j) * head_dim;

            float32x4_t _s0a = vdupq_n_f32(0.f);
            float32x4_t _s0b = vdupq_n_f32(0.f);
            float32x4_t _s0c = vdupq_n_f32(0.f);
            float32x4_t _s1a = vdupq_n_f32(0.f);
            float32x4_t _s1b = vdupq_n_f32(0.f);
            float32x4_t _s1c = vdupq_n_f32(0.f);
            float32x4_t _s2a = vdupq_n_f32(0.f);
            float32x4_t _s2b = vdupq_n_f32(0.f);
            float32x4_t _s2c = vdupq_n_f32(0.f);
            float32x4_t _s3a = vdupq_n_f32(0.f);
            float32x4_t _s3b = vdupq_n_f32(0.f);
            float32x4_t _s3c = vdupq_n_f32(0.f);
            float32x4_t _s4a = vdupq_n_f32(0.f);
            float32x4_t _s4b = vdupq_n_f32(0.f);
            float32x4_t _s4c = vdupq_n_f32(0.f);
            float32x4_t _s5a = vdupq_n_f32(0.f);
            float32x4_t _s5b = vdupq_n_f32(0.f);
            float32x4_t _s5c = vdupq_n_f32(0.f);
            float32x4_t _s6a = vdupq_n_f32(0.f);
            float32x4_t _s6b = vdupq_n_f32(0.f);
            float32x4_t _s6c = vdupq_n_f32(0.f);
            float32x4_t _s7a = vdupq_n_f32(0.f);
            float32x4_t _s7b = vdupq_n_f32(0.f);
            float32x4_t _s7c = vdupq_n_f32(0.f);

            for (int d = 0; d < head_dim; d++)
            {
                float32x4_t _qa = vld1q_f32(pQ);
                float32x4_t _qb = vld1q_f32(pQ + 4);
                float32x4_t _qc = vld1q_f32(pQ + 8);
                float32x4_t _ka = vld1q_f32(pK);
                float32x4_t _kb = vld1q_f32(pK + 4);
                _s0a = vfmaq_laneq_f32(_s0a, _qa, _ka, 0);
                _s0b = vfmaq_laneq_f32(_s0b, _qb, _ka, 0);
                _s0c = vfmaq_laneq_f32(_s0c, _qc, _ka, 0);
                _s1a = vfmaq_laneq_f32(_s1a, _qa, _ka, 1);
                _s1b = vfmaq_laneq_f32(_s1b, _qb, _ka, 1);
                _s1c = vfmaq_laneq_f32(_s1c, _qc, _ka, 1);
                _s2a = vfmaq_laneq_f32(_s2a, _qa, _ka, 2);
                _s2b = vfmaq_laneq_f32(_s2b, _qb, _ka, 2);
                _s2c = vfmaq_laneq_f32(_s2c, _qc, _ka, 2);
                _s3a = vfmaq_laneq_f32(_s3a, _qa, _ka, 3);
                _s3b = vfmaq_laneq_f32(_s3b, _qb, _ka, 3);
                _s3c = vfmaq_laneq_f32(_s3c, _qc, _ka, 3);
                _s4a = vfmaq_laneq_f32(_s4a, _qa, _kb, 0);
                _s4b = vfmaq_laneq_f32(_s4b, _qb, _kb, 0);
                _s4c = vfmaq_laneq_f32(_s4c, _qc, _kb, 0);
                _s5a = vfmaq_laneq_f32(_s5a, _qa, _kb, 1);
                _s5b = vfmaq_laneq_f32(_s5b, _qb, _kb, 1);
                _s5c = vfmaq_laneq_f32(_s5c, _qc, _kb, 1);
                _s6a = vfmaq_laneq_f32(_s6a, _qa, _kb, 2);
                _s6b = vfmaq_laneq_f32(_s6b, _qb, _kb, 2);
                _s6c = vfmaq_laneq_f32(_s6c, _qc, _kb, 2);
                _s7a = vfmaq_laneq_f32(_s7a, _qa, _kb, 3);
                _s7b = vfmaq_laneq_f32(_s7b, _qb, _kb, 3);
                _s7c = vfmaq_laneq_f32(_s7c, _qc, _kb, 3);
                pQ += 12;
                pK += 8;
            }

            float* pS = scoreT + j * 12;
            vst1q_f32(pS, _s0a);
            vst1q_f32(pS + 4, _s0b);
            vst1q_f32(pS + 8, _s0c);
            vst1q_f32(pS + 12, _s1a);
            vst1q_f32(pS + 16, _s1b);
            vst1q_f32(pS + 20, _s1c);
            vst1q_f32(pS + 24, _s2a);
            vst1q_f32(pS + 28, _s2b);
            vst1q_f32(pS + 32, _s2c);
            vst1q_f32(pS + 36, _s3a);
            vst1q_f32(pS + 40, _s3b);
            vst1q_f32(pS + 44, _s3c);
            vst1q_f32(pS + 48, _s4a);
            vst1q_f32(pS + 52, _s4b);
            vst1q_f32(pS + 56, _s4c);
            vst1q_f32(pS + 60, _s5a);
            vst1q_f32(pS + 64, _s5b);
            vst1q_f32(pS + 68, _s5c);
            vst1q_f32(pS + 72, _s6a);
            vst1q_f32(pS + 76, _s6b);
            vst1q_f32(pS + 80, _s6c);
            vst1q_f32(pS + 84, _s7a);
            vst1q_f32(pS + 88, _s7b);
            vst1q_f32(pS + 92, _s7c);

            if (maskptrs[0])
            {
                const int jn = n + j;
                sdpa_prefill_add_mask<T>(pS, maskptrs, jn, ni, std::min(8, key_seqlen - jn), 12);
            }
        }

        // online softmax over the valid keys of the block
        float32x4_t _alpha0;
        float32x4_t _alpha1;
        float32x4_t _alpha2;
        {
            float32x4_t _bmax0 = vdupq_n_f32(-FLT_MAX);
            float32x4_t _bmax1 = vdupq_n_f32(-FLT_MAX);
            float32x4_t _bmax2 = vdupq_n_f32(-FLT_MAX);
            for (int j = 0; j < max_jj; j++)
            {
                _bmax0 = vmaxq_f32(_bmax0, vld1q_f32(scoreT + j * 12));
                _bmax1 = vmaxq_f32(_bmax1, vld1q_f32(scoreT + j * 12 + 4));
                _bmax2 = vmaxq_f32(_bmax2, vld1q_f32(scoreT + j * 12 + 8));
            }

            float32x4_t _m_new0 = vmaxq_f32(_m0, _bmax0);
            float32x4_t _m_new1 = vmaxq_f32(_m1, _bmax1);
            float32x4_t _m_new2 = vmaxq_f32(_m2, _bmax2);

            // alpha is 0 for rows that have no history yet
            uint32x4_t _active0 = vmvnq_u32(vceqq_f32(_l0, vdupq_n_f32(0.f)));
            uint32x4_t _active1 = vmvnq_u32(vceqq_f32(_l1, vdupq_n_f32(0.f)));
            uint32x4_t _active2 = vmvnq_u32(vceqq_f32(_l2, vdupq_n_f32(0.f)));
            _alpha0 = exp_ps(vreinterpretq_f32_u32(vandq_u32(_active0, vreinterpretq_u32_f32(vsubq_f32(_m0, _m_new0)))));
            _alpha1 = exp_ps(vreinterpretq_f32_u32(vandq_u32(_active1, vreinterpretq_u32_f32(vsubq_f32(_m1, _m_new1)))));
            _alpha2 = exp_ps(vreinterpretq_f32_u32(vandq_u32(_active2, vreinterpretq_u32_f32(vsubq_f32(_m2, _m_new2)))));
            _alpha0 = vreinterpretq_f32_u32(vandq_u32(_active0, vreinterpretq_u32_f32(_alpha0)));
            _alpha1 = vreinterpretq_f32_u32(vandq_u32(_active1, vreinterpretq_u32_f32(_alpha1)));
            _alpha2 = vreinterpretq_f32_u32(vandq_u32(_active2, vreinterpretq_u32_f32(_alpha2)));

            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            for (int j = 0; j < max_jj; j++)
            {
                float32x4_t _p0 = exp_ps(vsubq_f32(vld1q_f32(scoreT + j * 12), _m_new0));
                float32x4_t _p1 = exp_ps(vsubq_f32(vld1q_f32(scoreT + j * 12 + 4), _m_new1));
                float32x4_t _p2 = exp_ps(vsubq_f32(vld1q_f32(scoreT + j * 12 + 8), _m_new2));
                vst1q_f32(scoreT + j * 12, _p0);
                vst1q_f32(scoreT + j * 12 + 4, _p1);
                vst1q_f32(scoreT + j * 12 + 8, _p2);
                _sum0 = vaddq_f32(_sum0, _p0);
                _sum1 = vaddq_f32(_sum1, _p1);
                _sum2 = vaddq_f32(_sum2, _p2);
            }

            _l0 = vfmaq_f32(_sum0, _l0, _alpha0);
            _l1 = vfmaq_f32(_sum1, _l1, _alpha1);
            _l2 = vfmaq_f32(_sum2, _l2, _alpha2);
            _m0 = _m_new0;
            _m1 = _m_new1;
            _m2 = _m_new2;
        }

        // pv, outT[value_dim_padded][12 rows]
        for (int g = 0; g < value_dim_padded / 8; g++)
        {
            float* pO = outT + g * 96;
            float32x4_t _o0a = vmulq_f32(vld1q_f32(pO), _alpha0);
            float32x4_t _o0b = vmulq_f32(vld1q_f32(pO + 4), _alpha1);
            float32x4_t _o0c = vmulq_f32(vld1q_f32(pO + 8), _alpha2);
            float32x4_t _o1a = vmulq_f32(vld1q_f32(pO + 12), _alpha0);
            float32x4_t _o1b = vmulq_f32(vld1q_f32(pO + 16), _alpha1);
            float32x4_t _o1c = vmulq_f32(vld1q_f32(pO + 20), _alpha2);
            float32x4_t _o2a = vmulq_f32(vld1q_f32(pO + 24), _alpha0);
            float32x4_t _o2b = vmulq_f32(vld1q_f32(pO + 28), _alpha1);
            float32x4_t _o2c = vmulq_f32(vld1q_f32(pO + 32), _alpha2);
            float32x4_t _o3a = vmulq_f32(vld1q_f32(pO + 36), _alpha0);
            float32x4_t _o3b = vmulq_f32(vld1q_f32(pO + 40), _alpha1);
            float32x4_t _o3c = vmulq_f32(vld1q_f32(pO + 44), _alpha2);
            float32x4_t _o4a = vmulq_f32(vld1q_f32(pO + 48), _alpha0);
            float32x4_t _o4b = vmulq_f32(vld1q_f32(pO + 52), _alpha1);
            float32x4_t _o4c = vmulq_f32(vld1q_f32(pO + 56), _alpha2);
            float32x4_t _o5a = vmulq_f32(vld1q_f32(pO + 60), _alpha0);
            float32x4_t _o5b = vmulq_f32(vld1q_f32(pO + 64), _alpha1);
            float32x4_t _o5c = vmulq_f32(vld1q_f32(pO + 68), _alpha2);
            float32x4_t _o6a = vmulq_f32(vld1q_f32(pO + 72), _alpha0);
            float32x4_t _o6b = vmulq_f32(vld1q_f32(pO + 76), _alpha1);
            float32x4_t _o6c = vmulq_f32(vld1q_f32(pO + 80), _alpha2);
            float32x4_t _o7a = vmulq_f32(vld1q_f32(pO + 84), _alpha0);
            float32x4_t _o7b = vmulq_f32(vld1q_f32(pO + 88), _alpha1);
            float32x4_t _o7c = vmulq_f32(vld1q_f32(pO + 92), _alpha2);

            const float* pS = scoreT;
            const float* pV = pv + g * group_stride + (size_t)n * 8;
            for (int j = 0; j < max_jj; j++)
            {
                float32x4_t _pa = vld1q_f32(pS);
                float32x4_t _pb = vld1q_f32(pS + 4);
                float32x4_t _pc = vld1q_f32(pS + 8);
                float32x4_t _va = vld1q_f32(pV);
                float32x4_t _vb = vld1q_f32(pV + 4);
                _o0a = vfmaq_laneq_f32(_o0a, _pa, _va, 0);
                _o0b = vfmaq_laneq_f32(_o0b, _pb, _va, 0);
                _o0c = vfmaq_laneq_f32(_o0c, _pc, _va, 0);
                _o1a = vfmaq_laneq_f32(_o1a, _pa, _va, 1);
                _o1b = vfmaq_laneq_f32(_o1b, _pb, _va, 1);
                _o1c = vfmaq_laneq_f32(_o1c, _pc, _va, 1);
                _o2a = vfmaq_laneq_f32(_o2a, _pa, _va, 2);
                _o2b = vfmaq_laneq_f32(_o2b, _pb, _va, 2);
                _o2c = vfmaq_laneq_f32(_o2c, _pc, _va, 2);
                _o3a = vfmaq_laneq_f32(_o3a, _pa, _va, 3);
                _o3b = vfmaq_laneq_f32(_o3b, _pb, _va, 3);
                _o3c = vfmaq_laneq_f32(_o3c, _pc, _va, 3);
                _o4a = vfmaq_laneq_f32(_o4a, _pa, _vb, 0);
                _o4b = vfmaq_laneq_f32(_o4b, _pb, _vb, 0);
                _o4c = vfmaq_laneq_f32(_o4c, _pc, _vb, 0);
                _o5a = vfmaq_laneq_f32(_o5a, _pa, _vb, 1);
                _o5b = vfmaq_laneq_f32(_o5b, _pb, _vb, 1);
                _o5c = vfmaq_laneq_f32(_o5c, _pc, _vb, 1);
                _o6a = vfmaq_laneq_f32(_o6a, _pa, _vb, 2);
                _o6b = vfmaq_laneq_f32(_o6b, _pb, _vb, 2);
                _o6c = vfmaq_laneq_f32(_o6c, _pc, _vb, 2);
                _o7a = vfmaq_laneq_f32(_o7a, _pa, _vb, 3);
                _o7b = vfmaq_laneq_f32(_o7b, _pb, _vb, 3);
                _o7c = vfmaq_laneq_f32(_o7c, _pc, _vb, 3);
                pS += 12;
                pV += 8;
            }

            vst1q_f32(pO, _o0a);
            vst1q_f32(pO + 4, _o0b);
            vst1q_f32(pO + 8, _o0c);
            vst1q_f32(pO + 12, _o1a);
            vst1q_f32(pO + 16, _o1b);
            vst1q_f32(pO + 20, _o1c);
            vst1q_f32(pO + 24, _o2a);
            vst1q_f32(pO + 28, _o2b);
            vst1q_f32(pO + 32, _o2c);
            vst1q_f32(pO + 36, _o3a);
            vst1q_f32(pO + 40, _o3b);
            vst1q_f32(pO + 44, _o3c);
            vst1q_f32(pO + 48, _o4a);
            vst1q_f32(pO + 52, _o4b);
            vst1q_f32(pO + 56, _o4c);
            vst1q_f32(pO + 60, _o5a);
            vst1q_f32(pO + 64, _o5b);
            vst1q_f32(pO + 68, _o5c);
            vst1q_f32(pO + 72, _o6a);
            vst1q_f32(pO + 76, _o6b);
            vst1q_f32(pO + 80, _o6c);
            vst1q_f32(pO + 84, _o7a);
            vst1q_f32(pO + 88, _o7b);
            vst1q_f32(pO + 92, _o7c);
        }
    }

    vst1q_f32(l, _l0);
    vst1q_f32(l + 4, _l1);
    vst1q_f32(l + 8, _l2);
}

// 8 query rows of one head against all keys of its kv head
// qT[head_dim][8] is prescaled, pk and pv are the packed key and value of the kv head
// maskptrs[8] are the mask rows of the 8 query rows, or all null
// ni is the number of valid query rows
// workspace: scoreT[block_n][8] outT[value_dim_padded][8]
// the result is outT[value_dim_padded][8] and l[8] (denominators)
template<typename T>
static void sdpa_prefill_attention_tile_8(const float* qT, const float* pk, const float* pv, const T* const* maskptrs, int ni, int head_dim, int value_dim_padded, int key_seqlen, int key_seqlen_padded, int block_n, float* scoreT, float* outT, float* l)
{
    const size_t group_stride = (size_t)key_seqlen_padded * 8;

    float32x4_t _m0 = vdupq_n_f32(-FLT_MAX);
    float32x4_t _m1 = vdupq_n_f32(-FLT_MAX);
    float32x4_t _l0 = vdupq_n_f32(0.f);
    float32x4_t _l1 = vdupq_n_f32(0.f);
    memset(outT, 0, value_dim_padded * 8 * sizeof(float));

    for (int n = 0; n < key_seqlen; n += block_n)
    {
        const int max_jj = std::min(key_seqlen - n, block_n);

        // qk, scoreT[j][8 rows]
        for (int j = 0; j < max_jj; j += 8)
        {
            const float* pQ = qT;
            const float* pK = pk + (size_t)(n + j) * head_dim;

            float32x4_t _s0a = vdupq_n_f32(0.f);
            float32x4_t _s0b = vdupq_n_f32(0.f);
            float32x4_t _s1a = vdupq_n_f32(0.f);
            float32x4_t _s1b = vdupq_n_f32(0.f);
            float32x4_t _s2a = vdupq_n_f32(0.f);
            float32x4_t _s2b = vdupq_n_f32(0.f);
            float32x4_t _s3a = vdupq_n_f32(0.f);
            float32x4_t _s3b = vdupq_n_f32(0.f);
            float32x4_t _s4a = vdupq_n_f32(0.f);
            float32x4_t _s4b = vdupq_n_f32(0.f);
            float32x4_t _s5a = vdupq_n_f32(0.f);
            float32x4_t _s5b = vdupq_n_f32(0.f);
            float32x4_t _s6a = vdupq_n_f32(0.f);
            float32x4_t _s6b = vdupq_n_f32(0.f);
            float32x4_t _s7a = vdupq_n_f32(0.f);
            float32x4_t _s7b = vdupq_n_f32(0.f);

            for (int d = 0; d < head_dim; d++)
            {
                float32x4_t _qa = vld1q_f32(pQ);
                float32x4_t _qb = vld1q_f32(pQ + 4);
                float32x4_t _ka = vld1q_f32(pK);
                float32x4_t _kb = vld1q_f32(pK + 4);
                _s0a = vfmaq_laneq_f32(_s0a, _qa, _ka, 0);
                _s0b = vfmaq_laneq_f32(_s0b, _qb, _ka, 0);
                _s1a = vfmaq_laneq_f32(_s1a, _qa, _ka, 1);
                _s1b = vfmaq_laneq_f32(_s1b, _qb, _ka, 1);
                _s2a = vfmaq_laneq_f32(_s2a, _qa, _ka, 2);
                _s2b = vfmaq_laneq_f32(_s2b, _qb, _ka, 2);
                _s3a = vfmaq_laneq_f32(_s3a, _qa, _ka, 3);
                _s3b = vfmaq_laneq_f32(_s3b, _qb, _ka, 3);
                _s4a = vfmaq_laneq_f32(_s4a, _qa, _kb, 0);
                _s4b = vfmaq_laneq_f32(_s4b, _qb, _kb, 0);
                _s5a = vfmaq_laneq_f32(_s5a, _qa, _kb, 1);
                _s5b = vfmaq_laneq_f32(_s5b, _qb, _kb, 1);
                _s6a = vfmaq_laneq_f32(_s6a, _qa, _kb, 2);
                _s6b = vfmaq_laneq_f32(_s6b, _qb, _kb, 2);
                _s7a = vfmaq_laneq_f32(_s7a, _qa, _kb, 3);
                _s7b = vfmaq_laneq_f32(_s7b, _qb, _kb, 3);
                pQ += 8;
                pK += 8;
            }

            float* pS = scoreT + j * 8;
            vst1q_f32(pS, _s0a);
            vst1q_f32(pS + 4, _s0b);
            vst1q_f32(pS + 8, _s1a);
            vst1q_f32(pS + 12, _s1b);
            vst1q_f32(pS + 16, _s2a);
            vst1q_f32(pS + 20, _s2b);
            vst1q_f32(pS + 24, _s3a);
            vst1q_f32(pS + 28, _s3b);
            vst1q_f32(pS + 32, _s4a);
            vst1q_f32(pS + 36, _s4b);
            vst1q_f32(pS + 40, _s5a);
            vst1q_f32(pS + 44, _s5b);
            vst1q_f32(pS + 48, _s6a);
            vst1q_f32(pS + 52, _s6b);
            vst1q_f32(pS + 56, _s7a);
            vst1q_f32(pS + 60, _s7b);

            if (maskptrs[0])
            {
                const int jn = n + j;
                sdpa_prefill_add_mask<T>(pS, maskptrs, jn, ni, std::min(8, key_seqlen - jn), 8);
            }
        }

        // online softmax over the valid keys of the block
        float32x4_t _alpha0;
        float32x4_t _alpha1;
        {
            float32x4_t _bmax0 = vdupq_n_f32(-FLT_MAX);
            float32x4_t _bmax1 = vdupq_n_f32(-FLT_MAX);
            for (int j = 0; j < max_jj; j++)
            {
                _bmax0 = vmaxq_f32(_bmax0, vld1q_f32(scoreT + j * 8));
                _bmax1 = vmaxq_f32(_bmax1, vld1q_f32(scoreT + j * 8 + 4));
            }

            float32x4_t _m_new0 = vmaxq_f32(_m0, _bmax0);
            float32x4_t _m_new1 = vmaxq_f32(_m1, _bmax1);

            // alpha is 0 for rows that have no history yet
            uint32x4_t _active0 = vmvnq_u32(vceqq_f32(_l0, vdupq_n_f32(0.f)));
            uint32x4_t _active1 = vmvnq_u32(vceqq_f32(_l1, vdupq_n_f32(0.f)));
            _alpha0 = exp_ps(vreinterpretq_f32_u32(vandq_u32(_active0, vreinterpretq_u32_f32(vsubq_f32(_m0, _m_new0)))));
            _alpha1 = exp_ps(vreinterpretq_f32_u32(vandq_u32(_active1, vreinterpretq_u32_f32(vsubq_f32(_m1, _m_new1)))));
            _alpha0 = vreinterpretq_f32_u32(vandq_u32(_active0, vreinterpretq_u32_f32(_alpha0)));
            _alpha1 = vreinterpretq_f32_u32(vandq_u32(_active1, vreinterpretq_u32_f32(_alpha1)));

            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            for (int j = 0; j < max_jj; j++)
            {
                float32x4_t _p0 = exp_ps(vsubq_f32(vld1q_f32(scoreT + j * 8), _m_new0));
                float32x4_t _p1 = exp_ps(vsubq_f32(vld1q_f32(scoreT + j * 8 + 4), _m_new1));
                vst1q_f32(scoreT + j * 8, _p0);
                vst1q_f32(scoreT + j * 8 + 4, _p1);
                _sum0 = vaddq_f32(_sum0, _p0);
                _sum1 = vaddq_f32(_sum1, _p1);
            }

            _l0 = vfmaq_f32(_sum0, _l0, _alpha0);
            _l1 = vfmaq_f32(_sum1, _l1, _alpha1);
            _m0 = _m_new0;
            _m1 = _m_new1;
        }

        // pv, outT[value_dim_padded][8 rows]
        for (int g = 0; g < value_dim_padded / 8; g++)
        {
            float* pO = outT + g * 64;
            float32x4_t _o0a = vmulq_f32(vld1q_f32(pO), _alpha0);
            float32x4_t _o0b = vmulq_f32(vld1q_f32(pO + 4), _alpha1);
            float32x4_t _o1a = vmulq_f32(vld1q_f32(pO + 8), _alpha0);
            float32x4_t _o1b = vmulq_f32(vld1q_f32(pO + 12), _alpha1);
            float32x4_t _o2a = vmulq_f32(vld1q_f32(pO + 16), _alpha0);
            float32x4_t _o2b = vmulq_f32(vld1q_f32(pO + 20), _alpha1);
            float32x4_t _o3a = vmulq_f32(vld1q_f32(pO + 24), _alpha0);
            float32x4_t _o3b = vmulq_f32(vld1q_f32(pO + 28), _alpha1);
            float32x4_t _o4a = vmulq_f32(vld1q_f32(pO + 32), _alpha0);
            float32x4_t _o4b = vmulq_f32(vld1q_f32(pO + 36), _alpha1);
            float32x4_t _o5a = vmulq_f32(vld1q_f32(pO + 40), _alpha0);
            float32x4_t _o5b = vmulq_f32(vld1q_f32(pO + 44), _alpha1);
            float32x4_t _o6a = vmulq_f32(vld1q_f32(pO + 48), _alpha0);
            float32x4_t _o6b = vmulq_f32(vld1q_f32(pO + 52), _alpha1);
            float32x4_t _o7a = vmulq_f32(vld1q_f32(pO + 56), _alpha0);
            float32x4_t _o7b = vmulq_f32(vld1q_f32(pO + 60), _alpha1);

            const float* pS = scoreT;
            const float* pV = pv + g * group_stride + (size_t)n * 8;
            for (int j = 0; j < max_jj; j++)
            {
                float32x4_t _pa = vld1q_f32(pS);
                float32x4_t _pb = vld1q_f32(pS + 4);
                float32x4_t _va = vld1q_f32(pV);
                float32x4_t _vb = vld1q_f32(pV + 4);
                _o0a = vfmaq_laneq_f32(_o0a, _pa, _va, 0);
                _o0b = vfmaq_laneq_f32(_o0b, _pb, _va, 0);
                _o1a = vfmaq_laneq_f32(_o1a, _pa, _va, 1);
                _o1b = vfmaq_laneq_f32(_o1b, _pb, _va, 1);
                _o2a = vfmaq_laneq_f32(_o2a, _pa, _va, 2);
                _o2b = vfmaq_laneq_f32(_o2b, _pb, _va, 2);
                _o3a = vfmaq_laneq_f32(_o3a, _pa, _va, 3);
                _o3b = vfmaq_laneq_f32(_o3b, _pb, _va, 3);
                _o4a = vfmaq_laneq_f32(_o4a, _pa, _vb, 0);
                _o4b = vfmaq_laneq_f32(_o4b, _pb, _vb, 0);
                _o5a = vfmaq_laneq_f32(_o5a, _pa, _vb, 1);
                _o5b = vfmaq_laneq_f32(_o5b, _pb, _vb, 1);
                _o6a = vfmaq_laneq_f32(_o6a, _pa, _vb, 2);
                _o6b = vfmaq_laneq_f32(_o6b, _pb, _vb, 2);
                _o7a = vfmaq_laneq_f32(_o7a, _pa, _vb, 3);
                _o7b = vfmaq_laneq_f32(_o7b, _pb, _vb, 3);
                pS += 8;
                pV += 8;
            }

            vst1q_f32(pO, _o0a);
            vst1q_f32(pO + 4, _o0b);
            vst1q_f32(pO + 8, _o1a);
            vst1q_f32(pO + 12, _o1b);
            vst1q_f32(pO + 16, _o2a);
            vst1q_f32(pO + 20, _o2b);
            vst1q_f32(pO + 24, _o3a);
            vst1q_f32(pO + 28, _o3b);
            vst1q_f32(pO + 32, _o4a);
            vst1q_f32(pO + 36, _o4b);
            vst1q_f32(pO + 40, _o5a);
            vst1q_f32(pO + 44, _o5b);
            vst1q_f32(pO + 48, _o6a);
            vst1q_f32(pO + 52, _o6b);
            vst1q_f32(pO + 56, _o7a);
            vst1q_f32(pO + 60, _o7b);
        }
    }

    vst1q_f32(l, _l0);
    vst1q_f32(l + 4, _l1);
}

// outT[value_dim_padded][MR] / l -> out rows i0 .. i0 + ni
template<typename TO>
static void sdpa_prefill_store_output(const float* outT, const float* l, Mat& top_blob_head, int i0, int ni, int MR, int value_dim)
{
    float inv_l[12];
    for (int i = 0; i < MR; i++)
    {
        inv_l[i] = l[i] == 0.f ? 0.f : 1.f / l[i];
    }

    TO* outptrs[12];
    for (int i = 0; i < ni; i++)
    {
        outptrs[i] = top_blob_head.row<TO>(i0 + i);
    }

    // full groups of 4 rows
    int r = 0;
    for (; r + 3 < ni; r += 4)
    {
        int d = 0;
        for (; d + 3 < value_dim; d += 4)
        {
            float32x4_t _r0 = vld1q_f32(outT + d * MR + r);
            float32x4_t _r1 = vld1q_f32(outT + (d + 1) * MR + r);
            float32x4_t _r2 = vld1q_f32(outT + (d + 2) * MR + r);
            float32x4_t _r3 = vld1q_f32(outT + (d + 3) * MR + r);
            transpose4x4_ps(_r0, _r1, _r2, _r3);
            sdpa_store4(outptrs[r] + d, vmulq_n_f32(_r0, inv_l[r]));
            sdpa_store4(outptrs[r + 1] + d, vmulq_n_f32(_r1, inv_l[r + 1]));
            sdpa_store4(outptrs[r + 2] + d, vmulq_n_f32(_r2, inv_l[r + 2]));
            sdpa_store4(outptrs[r + 3] + d, vmulq_n_f32(_r3, inv_l[r + 3]));
        }
        for (; d < value_dim; d++)
        {
            for (int i = r; i < r + 4; i++)
            {
                sdpa_store1(outptrs[i] + d, outT[d * MR + i] * inv_l[i]);
            }
        }
    }
    // remaining rows
    for (; r < ni; r++)
    {
        for (int d = 0; d < value_dim; d++)
        {
            sdpa_store1(outptrs[r] + d, outT[d * MR + r] * inv_l[r]);
        }
    }
}

template<typename T, typename TO>
static int sdpa_prefill(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int query_seqlen = query.h;
    const int num_heads = query.c;
    const int value_dim = value.w;
    const int key_seqlen = key.h;
    const int num_kv_heads = key.c;
    const int num_heads_per_kv_head = num_heads / num_kv_heads;

    const int key_seqlen_padded = (key_seqlen + 7) / 8 * 8;
    const int value_dim_padded = (value_dim + 7) / 8 * 8;

    // the kernels walk keys in panels of 8
    int block_n = sdpa_prefill_get_optimal_tile_n(head_dim, value_dim_padded, key_seqlen);
    block_n = std::max(8, (block_n + 7) / 8 * 8);

    // split query rows into tiles of 12 rows, the tail uses tiles of 8 rows,
    // at most 3 rows are padded unless there are fewer than 5 rows
    std::vector<int> tile_i0;
    std::vector<int> tile_mr;
    {
        int num_tiles12 = query_seqlen / 12;
        const int r = query_seqlen % 12;
        if (r > 0 && r <= 4 && num_tiles12 > 0)
            num_tiles12 -= 1;

        for (int k = 0; k < num_tiles12; k++)
        {
            tile_i0.push_back(k * 12);
            tile_mr.push_back(12);
        }

        int i0 = num_tiles12 * 12;
        while (i0 < query_seqlen)
        {
            const int remain = query_seqlen - i0;
            const int mr = remain > 8 && remain <= 12 ? 12 : 8;
            tile_i0.push_back(i0);
            tile_mr.push_back(mr);
            i0 += mr;
        }
    }
    const int num_tiles = (int)tile_i0.size();

    Mat packed_key(head_dim * key_seqlen_padded, 1, num_kv_heads, 4u, opt.workspace_allocator);
    if (packed_key.empty())
        return -100;

    Mat packed_value(value_dim_padded * key_seqlen_padded, 1, num_kv_heads, 4u, opt.workspace_allocator);
    if (packed_value.empty())
        return -100;

    #pragma omp parallel for num_threads(std::min(opt.num_threads, num_kv_heads * 2))
    for (int task_id = 0; task_id < num_kv_heads * 2; task_id++)
    {
        const int g = task_id / 2;
        if (task_id % 2 == 0)
            sdpa_prefill_pack_key<T>(key.channel(g), packed_key.channel(g), key_seqlen_padded);
        else
            sdpa_prefill_pack_value<T>(value.channel(g), packed_value.channel(g), key_seqlen_padded, value_dim_padded);
    }

    const int num_tasks = num_heads * num_tiles;
    const int nT = std::min(std::max(opt.num_threads, 1), num_tasks);

    // per thread: packed query, scores, output accumulator and denominators, sized for 12 rows
    const int workspace_size = 12 * head_dim + 12 * block_n + 12 * value_dim_padded + 12;
    Mat workspace(workspace_size, 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int q = task_id / num_tiles;
        const int i0 = tile_i0[task_id % num_tiles];
        const int MR = tile_mr[task_id % num_tiles];
        const int ni = std::min(MR, query_seqlen - i0);
        const int g = q / num_heads_per_kv_head;

        float* qT = workspace.channel(get_omp_thread_num());
        float* scoreT = qT + MR * head_dim;
        float* outT = scoreT + MR * block_n;
        float* l = outT + MR * value_dim_padded;

        sdpa_prefill_pack_query<T>(query.channel(q), qT, i0, ni, MR, scale);

        const T* maskptrs[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (!attn_mask_blob.empty())
        {
            const Mat mask_head = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0) : attn_mask_blob;
            for (int i = 0; i < ni; i++)
            {
                maskptrs[i] = mask_head.row<const T>(i0 + i);
            }
            // padded rows reuse a valid row, their results are dropped
            for (int i = ni; i < MR; i++)
            {
                maskptrs[i] = maskptrs[0];
            }
        }

        const float* pk = packed_key.channel(g);
        const float* pv = packed_value.channel(g);
        if (MR == 12)
            sdpa_prefill_attention_tile_12<T>(qT, pk, pv, maskptrs, ni, head_dim, value_dim_padded, key_seqlen, key_seqlen_padded, block_n, scoreT, outT, l);
        else
            sdpa_prefill_attention_tile_8<T>(qT, pk, pv, maskptrs, ni, head_dim, value_dim_padded, key_seqlen, key_seqlen_padded, block_n, scoreT, outT, l);

        Mat top_blob_head = top_blob.channel(q);
        sdpa_prefill_store_output<TO>(outT, l, top_blob_head, i0, ni, MR, value_dim);
    }

    return 0;
}

#endif // __aarch64__
