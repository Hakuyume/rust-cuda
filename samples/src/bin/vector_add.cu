// original: cuda/samples/0_Simple/vectorAdd

extern "C" __global__ void vector_add(const float *a, const float *b, float *c, size_t num)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num)
    c[i] = a[i] + b[i];
}
