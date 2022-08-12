#include "math.h"
#include "fft_hls.hpp"

unsigned int reverse_bits(unsigned int input) {
	int i, rev = 0;
	for (i = 0; i < M; i++) {
		rev = (rev << 1) | (input & 1);
		input = input >> 1;
	}
	return rev;
}

void bit_reverse(DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		 DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE]) {
  unsigned int reversed;
  unsigned int i;


  for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE II=2
	  reversed = reverse_bits(i); // Find the bit reversed index
		if (i <= reversed) {
			// Swap the real values
			OUT_R[i] = X_R[reversed];
			OUT_R[reversed] = X_R[i];

			// Swap the imaginary values
			OUT_I[i] = X_I[reversed];
			OUT_I[reversed] = X_I[i];
		}
	}
}
void fft_stage_1( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 1;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {

#pragma HLS PIPELINE II=6
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }

		}
}
void fft_stage_2( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 2;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=6
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_3( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 3;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=6
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_4( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 4;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_5( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 5;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_6( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 6;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_7( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 7;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_8( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 8;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT

	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128

		// Compute butterflies that use same W**k
		dft_loop:
		for (int i = j; i < SIZE; i += DFTpts) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = i + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[i] - temp_R;
		  Out_I[i_lower] = X_I[i] - temp_I;
		  Out_R[i] = X_R[i] + temp_R;
		  Out_I[i] = X_I[i] + temp_I;
		  }
		}
}
void fft_stage_9( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 9;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
#pragma HLS PIPELINE II=10
		DTYPE c = cos_coefficients_table[j<<(10-stage)];
		DTYPE s = sin_coefficients_table[j<<(10-stage)];
		int i = j;
		int i_lower = i + numBF; // index of lower point in butterfly
		int i1 = i + DFTpts;
		int i1_lower = i1 + numBF; // index of lower point in butterfly

		DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;

		Out_R[i_lower] = X_R[i] - temp_R;
		Out_I[i_lower] = X_I[i] - temp_I;
		Out_R[i] = X_R[i] + temp_R;
		Out_I[i] = X_I[i] + temp_I;

		DTYPE temp1_R = X_R[i1_lower] * c - X_I[i1_lower] * s;
		DTYPE temp1_I = X_I[i1_lower] * c + X_R[i1_lower] * s;
		Out_R[i1_lower] = X_R[i1] - temp1_R;
		Out_I[i1_lower] = X_I[i1] - temp1_I;
		Out_R[i1] = X_R[i1] + temp1_R;
		Out_I[i1] = X_I[i1] + temp1_I;
		}

}
void fft_stage_10( DTYPE X_R[SIZE], DTYPE X_I[SIZE],
		     DTYPE Out_R[SIZE], DTYPE Out_I[SIZE]) {
	int stage = 10;
	int DFTpts = 1 << stage;    // DFT = 2^stage = points in sub DFT
	int numBF = DFTpts / 2;     // Butterfly WIDTHS in sub-DFT
	DTYPE e = -6.283185307178 / DFTpts;
	// Perform butterflies for j-th stage
	butterfly_loop:
	for (int j = 0; j < numBF; j++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
		// Compute butterflies that use same W**k
		dft_loop:
#pragma HLS PIPELINE II=6

			DTYPE c = cos_coefficients_table[j<<(10-stage)];
			DTYPE s = sin_coefficients_table[j<<(10-stage)];
		  int i_lower = j + numBF; // index of lower point in butterfly
		  DTYPE temp_R = X_R[i_lower] * c - X_I[i_lower] * s;
		  DTYPE temp_I = X_I[i_lower] * c + X_R[i_lower] * s;
		  Out_R[i_lower] = X_R[j] - temp_R;
		  Out_I[i_lower] = X_I[j] - temp_I;
		  Out_R[j] = X_R[j] + temp_R;
		  Out_I[j] = X_I[j] + temp_I;

	}
}


void fft_hls(DTYPE X_R[SIZE], DTYPE X_I[SIZE], DTYPE OUT_R[SIZE], DTYPE OUT_I[SIZE])
{
#pragma HLS INTERFACE m_axi port=OUT_R offset=slave bundle=out depth=1024
#pragma HLS INTERFACE m_axi port=OUT_I offset=slave bundle=out depth=1024
#pragma HLS INTERFACE m_axi port=X_R offset=slave bundle=in depth=1024
#pragma HLS INTERFACE m_axi port=X_I offset=slave bundle=in depth=1024
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS DATAFLOW
	DTYPE	Stage1_R[SIZE], Stage1_I[SIZE];
	DTYPE	Stage2_R[SIZE], Stage2_I[SIZE];
	DTYPE	Stage3_R[SIZE], Stage3_I[SIZE];
	DTYPE	Stage4_R[SIZE], Stage4_I[SIZE];
	DTYPE	Stage5_R[SIZE], Stage5_I[SIZE];
	DTYPE	Stage6_R[SIZE], Stage6_I[SIZE];
	DTYPE	Stage7_R[SIZE], Stage7_I[SIZE];
	DTYPE	Stage8_R[SIZE], Stage8_I[SIZE];
	DTYPE	Stage9_R[SIZE], Stage9_I[SIZE];
	DTYPE	Stage10_R[SIZE], Stage10_I[SIZE];

	bit_reverse(X_R, X_I, Stage1_R, Stage1_I);
	fft_stage_1(Stage1_R, Stage1_I, Stage2_R, Stage2_I);
	fft_stage_2(Stage2_R, Stage2_I, Stage3_R, Stage3_I);
	fft_stage_3(Stage3_R, Stage3_I, Stage4_R, Stage4_I);
	fft_stage_4(Stage4_R, Stage4_I, Stage5_R, Stage5_I);
	fft_stage_5(Stage5_R, Stage5_I, Stage6_R, Stage6_I);
	fft_stage_6(Stage6_R, Stage6_I, Stage7_R, Stage7_I);
	fft_stage_7(Stage7_R, Stage7_I, Stage8_R, Stage8_I);
	fft_stage_8(Stage8_R, Stage8_I, Stage9_R, Stage9_I);
	fft_stage_9(Stage9_R, Stage9_I, Stage10_R, Stage10_I);
	fft_stage_10(Stage10_R, Stage10_I, OUT_R, OUT_I);
}

