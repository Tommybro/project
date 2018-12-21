__kernel void kernel_conv(__global float *in, __global float *out, __global float *weight, __global float *bias, int H, int W, int C, int stride) {
	int HOUT = H / stride;
	int WOUT = W / stride;

	int k = get_global_id(2);
	int hout = get_global_id(1);
	int wout = get_global_id(0);

	float sum = bias[k];
	for (int c = 0; c < C; ++c) {
		for (int r = 0; r < 3; ++r) {
			for (int s = 0; s < 3; ++s) {
				int h = hout * stride + r - 1;
				int w = wout * stride + s - 1;
				if (!(h < 0 || h >= H || w < 0 || w >= W)) {
					sum += in[c * H * W + h * W + w] * weight[k * C * 3 * 3 + c * 3 * 3 + r * 3 + s];
				}
			}
		}
	}

	out[k * HOUT * WOUT + hout * WOUT + wout] = sum;
}

__kernel void kernel_conv_relu(__global float *in, __global float *out, __global float *weight, __global float *bias, int H, int W, int C, int stride) {
	int HOUT = H / stride;
	int WOUT = W / stride;

	int k = get_global_id(2);
	int hout = get_global_id(1);
	int wout = get_global_id(0);

	float sum = bias[k];
	for (int c = 0; c < C; ++c) {
		for (int r = 0; r < 3; ++r) {
			for (int s = 0; s < 3; ++s) {
				int h = hout * stride + r - 1;
				int w = wout * stride + s - 1;
				if (!(h < 0 || h >= H || w < 0 || w >= W)) {
					sum += in[c * H * W + h * W + w] * weight[k * C * 3 * 3 + c * 3 * 3 + r * 3 + s];
				}
			}
		}
	}

	out[k * HOUT * WOUT + hout * WOUT + wout] = sum > 0 ? sum : 0;
}

__kernel void kernel_fc(__global float *in, __global float *out, __global float *weight,
                        __global float *bias, int C) {
    int k = get_global_id(0);

    float s = bias[k];
    for (int c = 0; c < C; ++c) {
        s += in[c] * weight[k * C + c];
    }

    out[k] = s > 0 ? s : 0;
}

__kernel void kernel_fuse1(__global float *ml, __global float *out) {
    int k = get_global_id(2);
    int h = get_global_id(1);
    int w = get_global_id(0);

    out[k * 28 * 28 + h * 28 + w] = ml[k * 28 * 28 + h * 28 + w];
}

__kernel void kernel_fuse2(__global float *gf, __global float *out) {
    int k = get_global_id(2) + 256;
    int h = get_global_id(1);
    int w = get_global_id(0);

    out[k * 28 * 28 + h * 28 + w] = gf[k - 256];
}

__kernel void kernel_upsample(__global float *in, __global float *out, int H, int W) {
    int c = get_global_id(2);
    int h = get_global_id(1);
    int w = get_global_id(0);

    float t = in[c * H * W + h * W + w];
    out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 0)] = t;
    out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 1)] = t;
    out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 0)] = t;
    out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 1)] = t;
}
