/*
 * Placeholder OpenCL kernel
 */

__kernel void bitonic(__global unsigned int *data, int kStart, int kEnd, int jStart, int jEnd)
{ 
	int i, j, k, ixj;
	unsigned int tmp;
	
	i = get_global_id(0);
	
	for(k = kStart; k <= kEnd; k = k << 1) {
		for(j = jStart; j > jEnd; j = j >> 1) {
			ixj = i^j;
			if( ixj > i ){
				if( (i&k) == 0 && data[i]>data[ixj]) {
					tmp = data[i];
					data[i] = data[ixj];
					data[ixj] = tmp;
				}
				if( (i&k) != 0 && data[i]<data[ixj]) {
					tmp = data[i];
					data[i] = data[ixj];
					data[ixj] = tmp;
				}
			}
			jStart = k;
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
	}
}
