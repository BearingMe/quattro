// M05 elementary compute operation: float32 vector add.
// Deliberately trivial — the milestone validates the harness (upload, dispatch,
// readback, sync, timestamps, diagnostics), not the arithmetic.

StructuredBuffer<float> g_a   : register(t0);
StructuredBuffer<float> g_b   : register(t1);
RWStructuredBuffer<float> g_out : register(u0);

cbuffer Params : register(b0)
{
    uint g_count;
};

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i < g_count)
    {
        g_out[i] = g_a[i] + g_b[i];
    }
}
