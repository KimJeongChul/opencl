__kernel void mat_mul(__global float *A, 
	__global float *B, 
	__global float *C,
	int ROW_A, 
	int COL_A, 
	int COL_B) {

	int i = get_global_id(1);
	int j = get_global_id(0);
	int k;
	float sum = 0.0f;

	if( i < ROW_A && j < COL_B) {
		for(k = 0; k < COL_A; k++) {
			sum += A[i * COL_A + k] * B[k * COL_B + j];
		}
		C[i * COL_B + j] = sum;
	}
}