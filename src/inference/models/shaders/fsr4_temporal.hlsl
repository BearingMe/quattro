// M13 buffer-based temporal adapter, derived from the pinned FSR 4.0.2
// pre_common/post_common/mlsr_optimized_includes algorithms. This is NOT the
// official texture-sampling implementation; filtering equivalence is unverified.
// Selected permutation: linear/manual-exposure/LR-MV/inverted-depth/no-RCAS.
#define FFX_MLSR_COLORSPACE 0
#include "ffx_fsr4_gpu_colorspace_common.h"

ByteAddressBuffer src : register(t0);
ByteAddressBuffer aux : register(t1);
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer reprojection : register(u1);

static const uint W = 1920, H = 1080;

float4 Half4(ByteAddressBuffer data, uint offset) {
    const uint2 words = data.Load2(offset);
    return float4(f16tof32(words.x & 65535), f16tof32(words.x >> 16),
                  f16tof32(words.y & 65535), f16tof32(words.y >> 16));
}
uint2 PackHalf4(float4 v) {
    return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16),
                 f32tof16(v.z) | (f32tof16(v.w) << 16));
}

// The scene buffer header matches FrameConstants; records match FramePixel.
#if TEMPORAL_PRE
#define scene src
#else
#define scene aux
#endif

uint2 RenderSize() { return scene.Load2(8); }
float2 Jitter() { return asfloat(scene.Load2(16)); }
float Exposure() {
    const float e = asfloat(scene.Load(24));
    return (e == 0 ? 1 : e) / asfloat(scene.Load(28));
}
float3 Color(int2 p) {
    p = clamp(p, int2(0, 0), int2(RenderSize()) - 1);
    return asfloat(scene.Load3(64 + (uint(p.y) * RenderSize().x + uint(p.x)) * 24));
}
float Depth(int2 p) {
    if (any(p < 0) || any(p >= int2(RenderSize())))
        return 0; // Texture2D.Load OOB behavior, not color's clamp behavior.
    return asfloat(scene.Load(64 + (uint(p.y) * RenderSize().x + uint(p.x)) * 24 + 12));
}
float2 Velocity(int2 p) {
    if (any(p < 0) || any(p >= int2(RenderSize())))
        return 0;
    float2 v = asfloat(scene.Load2(64 + (uint(p.y) * RenderSize().x + uint(p.x)) * 24 + 16));
    v *= asfloat(scene.Load2(32));
    if (scene.Load(48) & 4)
        v -= asfloat(scene.Load2(40));
    return v;
}
float3 YCbCr(float3 rgb) {
    const float y = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    return float3(y, (rgb.b - y) / 1.8556, (rgb.r - y) / 1.5748);
}
float Gaussian(float d, float scale) {
    const float x = d * scale;
    return exp(x * x * (-0.5 / (0.47 * 0.47)));
}

#if TEMPORAL_PRE
float4 Upscale(uint2 p) {
    const float2 invScale = float2(RenderSize()) / float2(W, H);
    const float2 position = -0.5 + invScale * 0.5 + float2(p) * invScale + Jitter();
    const int2 base = int2(floor(position));
    const float2 scale = 1.0 / invScale;
    float total = 0, center = 0;
    float3 color = 0;
    [unroll] for (int y = 0; y < 2; ++y) {
        [unroll] for (int x = 0; x < 2; ++x) {
            const int2 tap = base + int2(x, y);
            const float weight = Gaussian(abs(tap.x - position.x), scale.x) *
                                 Gaussian(abs(tap.y - position.y), scale.y);
            color += ApplyMuLaw(Color(tap) * Exposure()) * weight;
            total += weight;
            center = max(center, weight);
        }
    }
    return float4(color / total, center);
}
int2 Dilate(uint2 p) {
    const float2 invScale = float2(RenderSize()) / float2(W, H);
    const float2 origin = 0.5 * invScale - 0.5;
    int2 selected = int2(round(float2(p) * invScale + origin));
    float nearest = Depth(selected);
    // Preserve the provider's neighbour order and strict depth comparison.
    [unroll] for (int x = -1; x <= 1; ++x) {
        [unroll] for (int y = -1; y <= 1; ++y) {
            if (x == 0 && y == 0) continue;
            const int2 tap = int2(round((float2(p) + float2(x, y)) * invScale + origin));
            const float depth = Depth(tap);
            if (depth > nearest) { nearest = depth; selected = tap; }
        }
    }
    return selected;
}
float4 HistoryPoint(int2 p, uint offset) {
    p = clamp(p, int2(0, 0), int2(W - 1, H - 1));
    return Half4(aux, (uint(p.y) * W + uint(p.x)) * 32 + offset);
}
float4 HistoryLinear(float2 p, uint offset) {
    const int2 base = int2(floor(p));
    const float2 f = p - base;
    return lerp(lerp(HistoryPoint(base, offset), HistoryPoint(base + int2(1, 0), offset), f.x),
                lerp(HistoryPoint(base + int2(0, 1), offset), HistoryPoint(base + 1, offset), f.x), f.y);
}
float3 HistoryCubic(float2 p) {
    // Nine-fetch Catmull-Rom form from the reference. Software bilinear is
    // explicit here; fixed-function sampler precision is NOT claimed exact.
    const float2 base = floor(p), f = p - base;
    const float2 w0 = f * (-0.5 + f * (1 - 0.5 * f));
    const float2 w1 = 1 + f * f * (-2.5 + 1.5 * f);
    const float2 w2 = f * (0.5 + f * (2 - 1.5 * f));
    const float2 w3 = f * f * (-0.5 + 0.5 * f);
    const float2 w12 = w1 + w2, middle = base + w2 / w12;
    float3 value = 0;
    value += HistoryLinear(base - 1, 0).rgb * w0.x * w0.y;
    value += HistoryLinear(float2(middle.x, base.y - 1), 0).rgb * w12.x * w0.y;
    value += HistoryLinear(float2(base.x + 2, base.y - 1), 0).rgb * w3.x * w0.y;
    value += HistoryLinear(float2(base.x - 1, middle.y), 0).rgb * w0.x * w12.y;
    value += HistoryLinear(middle, 0).rgb * w12.x * w12.y;
    value += HistoryLinear(float2(base.x + 2, middle.y), 0).rgb * w3.x * w12.y;
    value += HistoryLinear(float2(base.x - 1, base.y + 2), 0).rgb * w0.x * w3.y;
    value += HistoryLinear(float2(middle.x, base.y + 2), 0).rgb * w12.x * w3.y;
    value += HistoryLinear(base + 2, 0).rgb * w3.x * w3.y;
    return value;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= W * H) return;
    const uint2 p = uint2(tid.x % W, tid.x / W);
    const float4 current = Upscale(p);
    float2 velocity = Velocity(Dilate(p));
    const bool invalid = any(isnan(velocity)) || any(isinf(velocity)) || any(velocity > 1000000);
    if (invalid) velocity = 0;
    const bool reset = scene.Load(0) != 0;
    float3 history = reset ? current.rgb : float3(0, 0, 0);
    float4 recurrent = 0;
    if (!reset && !invalid) {
        const float2 previous = float2(p) + velocity * float2(W, H);
        if (all(previous >= 0) && all(previous <= float2(W - 1, H - 1))) {
            history = HistoryCubic(previous);
            recurrent = HistoryLinear(previous, 8);
        }
        history = max(history, 0);
        const float weight = current.a * 0.1;
        history = weight * current.rgb + (1 - weight) * history;
    }
    // Resource conversion to FP16 occurs before postprocessing reads history.
    reprojection.Store2(tid.x * 8, PackHalf4(float4(history, 0)));
    const float3 a = YCbCr(current.rgb), b = YCbCr(history);
    const float distance = sqrt(dot(a.yz - b.yz, a.yz - b.yz) + 1e-6);
    const half4 first = half4(2 * half(a.x) - 1, 2 * half(b.x) - 1,
                              half(distance), 2 * half(recurrent.x) - 1);
    const half4 second = half4(2 * half(recurrent.y) - 1, 2 * half(recurrent.z) - 1,
                               2 * half(recurrent.w) - 1, 0);
    dst.Store2(tid.x * 16, PackHalf4(float4(first)));
    dst.Store2(tid.x * 16 + 8, PackHalf4(float4(second)));
}
#else
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= W * H) return;
    const uint2 p = uint2(tid.x % W, tid.x / W);
    const float4 filter = Half4(src, tid.x * 16);
    const half4 rawRecurrent = half4(Half4(src, tid.x * 16 + 8));
    const half4 recurrent = 1.0h / (1.0h + exp(-rawRecurrent));
    const float positive = exp(filter.x), negative = exp(-filter.x);
    const float correlation = (positive - negative) / (positive + negative);
    const float sx = 2.0 / (1 + exp(-filter.y));
    const float sy = 2.0 / (1 + exp(-filter.z));
    const float2 invScale = float2(RenderSize()) / float2(W, H);
    const float2 position = -0.5 + invScale * 0.5 + float2(p) * invScale + Jitter();
    const int2 base = int2(round(position));
    float3 color = 0;
    float total = 0;
    [unroll] for (int y = -1; y <= 1; ++y) {
        [unroll] for (int x = -1; x <= 1; ++x) {
            const int2 tap = base + int2(x, y);
            const float2 distance = (float2(tap) - position) / invScale;
            const float weight = exp((-0.5 / (0.47 * 0.47)) *
                (distance.x * distance.x * sx * sx + 2 * correlation * distance.x * distance.y * sx * sy +
                 distance.y * distance.y * sy * sy));
            color += ApplyMuLaw(Color(tap) * Exposure()) * weight;
            total += weight;
        }
    }
    const uint2 historyWords = reprojection.Load2(tid.x * 8);
    const float3 history = float3(f16tof32(historyWords.x & 65535),
                                  f16tof32(historyWords.x >> 16), f16tof32(historyWords.y & 65535));
    const float blend = 1.0 / (1 + exp(-filter.w));
    const float3 modelColor = color / total * (1 - blend) + history * blend;
    const float3 finalColor = clamp(RemoveMuLaw(modelColor) / Exposure(), 0, 64000);
    dst.Store2(tid.x * 32, PackHalf4(float4(modelColor, 0)));
    dst.Store2(tid.x * 32 + 8, PackHalf4(float4(recurrent)));
    dst.Store4(tid.x * 32 + 16, asuint(float4(finalColor, 1)));
}
#endif
