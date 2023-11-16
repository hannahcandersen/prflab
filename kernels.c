/********************************************************
 * Kernels to be optimized for the OS&C prflab.
 * Acknowledgment: This lab is an extended version of the
 * CS:APP Performance Lab
 ********************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "smooth.h" // helper functions for naive_smooth
#include "blend.h"  // helper functions for naive_blend
#include <limits.h>
#include <immintrin.h>

/*
 * Please fill in the following struct
 */
student_t student = {
    "hcan",               /* ITU alias */
    "Hannah C. Andersen", /* Full name */
    "hcan@itu.dk",        /* Email address */
};

/******************************************************************************
 * ROTATE KERNEL
 *****************************************************************************/

// Your different versions of the rotate kernel go here

/*
 * naive_rotate - The naive baseline version of rotate
 */
/* stride pattern, visualization (we recommend that you draw this for your functions):
    dst         src
    3 7 B F     0 1 2 3
    2 6 A E     4 5 6 7
    1 5 9 D     8 9 A B
    0 4 8 C     C D E F
 */

char naive_rotate_descr[] = "naive_rotate: Naive baseline implementation";
void naive_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            dst[RIDX(dim - 1 - j, i, dim)] = src[RIDX(i, j, dim)];
}

/*
 * rotate - Your current working version of rotate
 * IMPORTANT: This is the version you will be graded on
 */
// row i -> column i
// column j -> row dim - 1 - j
char rotate_descr[] = "rotate: Current working version";
// dim is number of columns (N)
void rotate(int dim, pixel *src, pixel *dst) 
{
    register int i, j; // register keyword: save variable in register instead of memory
    register int dim_minus_1 = dim - 1;

    for (i = 0; i < dim; i += 4)
    { // Unrolling by 4

        for (j = 0; j < dim; j += 4)
        // Unrolling by 4
        {
            // Inlining RIDX macro because this can give overhead

            // Precompute indices
            int dim_minus_1_minus_j = dim_minus_1 - j;

            // First row (from i) and first four columns (j) in this row
            dst[(dim_minus_1_minus_j)*dim + i] = src[i * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i] = src[i * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i] = src[i * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i] = src[i * dim + j + 3];

            // Second row (i + 1) and first four columns (j) in this row
            dst[(dim_minus_1_minus_j)*dim + i + 1] = src[(i + 1) * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i + 1] = src[(i + 1) * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i + 1] = src[(i + 1) * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i + 1] = src[(i + 1) * dim + j + 3];

            // Third row (i + 2) and first four columns (j) in this row
            dst[(dim_minus_1_minus_j)*dim + i + 2] = src[(i + 2) * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i + 2] = src[(i + 2) * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i + 2] = src[(i + 2) * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i + 2] = src[(i + 2) * dim + j + 3];

            // Fourth row (i + 3) and first four columns (j) in this row
            dst[(dim_minus_1_minus_j)*dim + i + 3] = src[(i + 3) * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i + 3] = src[(i + 3) * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i + 3] = src[(i + 3) * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i + 3] = src[(i + 3) * dim + j + 3];
        }
    }
}

/*
 * register_rotate_functions - Register all of your different versions
 *     of the rotate kernel with the driver by calling the
 *     add_rotate_function() for each test function.
 */
void register_rotate_functions()
{
    add_rotate_function(&rotate, rotate_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * ROTATE_T KERNEL
 *****************************************************************************/

// Your different versions of the rotate_t kernel go here
// (i.e. rotate with multi-threading)


/*
 * rotate_t - Your current working version of rotate_t
 * IMPORTANT: This is the version you will be graded on
 */

// Struct for thread arguments
typedef struct
{
    int dim;
    pixel *src;
    pixel *dst;
    int start_row;
    int end_row;
} thread_args;

// Worker thread function
void *rotate_thread(void *arg)
{
    thread_args *args = (thread_args *)arg;

    int dim = args->dim;
    pixel *src = args->src;
    pixel *dst = args->dst;
    int start_row = args->start_row;
    int end_row = args->end_row;

    register int i, j;
    register int dim_minus_1 = dim - 1;

    // Same code as in rotate
    for (i = start_row; i < end_row; i += 4)
    { // Unrolling by 4

        for (j = 0; j < dim; j += 4)
        // Unrolling by 4
        {
            // Inlining RIDX macro because this can give overhead

            // Precompute indices
            int dim_minus_1_minus_j = dim_minus_1 - j;

            // First row
            dst[(dim_minus_1_minus_j)*dim + i] = src[i * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i] = src[i * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i] = src[i * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i] = src[i * dim + j + 3];

            // Second row
            dst[(dim_minus_1_minus_j)*dim + i + 1] = src[(i + 1) * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i + 1] = src[(i + 1) * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i + 1] = src[(i + 1) * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i + 1] = src[(i + 1) * dim + j + 3];

            // Third row
            dst[(dim_minus_1_minus_j)*dim + i + 2] = src[(i + 2) * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i + 2] = src[(i + 2) * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i + 2] = src[(i + 2) * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i + 2] = src[(i + 2) * dim + j + 3];

            // Fourth row
            dst[(dim_minus_1_minus_j)*dim + i + 3] = src[(i + 3) * dim + j];
            dst[(dim_minus_1_minus_j - 1) * dim + i + 3] = src[(i + 3) * dim + j + 1];
            dst[(dim_minus_1_minus_j - 2) * dim + i + 3] = src[(i + 3) * dim + j + 2];
            dst[(dim_minus_1_minus_j - 3) * dim + i + 3] = src[(i + 3) * dim + j + 3];
        }
    }
    return NULL; // return value is not used
}

// Main function
// 4 threads because 4 cores
char rotate_t_descr[] = "rotate_t: Multithreaded version";
void rotate_t(int dim, pixel *src, pixel *dst)
{
    // This method is only good for large input
    // Check if input is small enough that it is better to call rotate
    if (dim < 512)
    {
        rotate(dim, src, dst); // Call rotate instead
        return; 
    }

    pthread_t threads[4]; // 4 threads because 4 cores
    thread_args thread_data[4]; // Thread arguments

    int rows_per_thread = dim / 4; // Divide rows in 4 parts

    for (int t = 0; t < 4; t++)
    {
        // create thread arguments
        thread_data[t].dim = dim;
        thread_data[t].src = src;
        thread_data[t].dst = dst;
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t == 3) ? dim : (t + 1) * rows_per_thread; // Last thread should handle the rest of the rows

        pthread_create(&threads[t], NULL, rotate_thread, (void *)&thread_data[t]); // Create thread
    }

    for (int t = 0; t < 4; t++)
    {
        pthread_join(threads[t], NULL); // Wait for all threads to finish
    }
}

/*********************************************************************
 * register_rotate_t_functions - Register all of your different versions
 *     of the rotate_t kernel with the driver by calling the
 *     add_rotate_t_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.
 *********************************************************************/

void register_rotate_t_functions()
{
    add_rotate_t_function(&rotate_t, rotate_t_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND KERNEL
 *****************************************************************************/

// Your different versions of the blend kernel go here.

char naive_blend_descr[] = "naive_blend: Naive baseline implementation";
void naive_blend(int dim, pixel *src, pixel *dst) // reads global variable `pixel bgc`
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            blend_pixel(&src[RIDX(i, j, dim)], &dst[RIDX(i, j, dim)], &bgc); // `blend_pixel` defined in blend.c
}

char blend_descr[] = "blend: Current working version"; // IMPORTANT: The version you will be graded on
void blend(int dim, pixel *src, pixel *dst) 
{
    register int i; // Tell compiler to save variable in register

    // Move code from blend_pixel into this function to avoid function call overhead
    // Only use one loop instead of two because the picture is represented as a 1D array. 
        // Each pixel can be processed independently of the others so we can use only one loop
    // Use loop unrolling to handle four pixels at a time
    for (i = 0; i < dim * dim; i += 4)
    {
        // First pixel
        pixel *s1 = &src[i];
        pixel *d1 = &dst[i];
        pixel *b1 = &bgc;
        float a1 = ((float)(s1->alpha)) / USHRT_MAX; // Alpha value

        // Second pixel
        pixel *s2 = &src[i + 1];
        pixel *d2 = &dst[i + 1];
        pixel *b2 = &bgc;
        float a2 = ((float)(s2->alpha)) / USHRT_MAX;

        // Third pixel
        pixel *s3 = &src[i + 2];
        pixel *d3 = &dst[i + 2];
        pixel *b3 = &bgc;
        float a3 = ((float)(s3->alpha)) / USHRT_MAX;

        // Fourth pixel
        pixel *s4 = &src[i + 3];
        pixel *d4 = &dst[i + 3];
        pixel *b4 = &bgc;
        float a4 = ((float)(s4->alpha)) / USHRT_MAX;

        // Blend
        d1->red = (a1 * s1->red) + ((1 - a1) * b1->red);       // Access destinations red property and set it to the blend of source and background red colours
        d1->green = (a1 * s1->green) + ((1 - a1) * b1->green); // Access destinations green property and set it to the blend of source and background green colours
        d1->blue = (a1 * s1->blue) + ((1 - a1) * b1->blue);    // Access destinations blue property and set it to the blend of source and background blue colours
        d1->alpha = USHRT_MAX;                                 // Make destination's alpha fully opaque

        d2->red = (a2 * s2->red) + ((1 - a2) * b2->red);
        d2->green = (a2 * s2->green) + ((1 - a2) * b2->green);
        d2->blue = (a2 * s2->blue) + ((1 - a2) * b2->blue);
        d2->alpha = USHRT_MAX;

        d3->red = (a3 * s3->red) + ((1 - a3) * b3->red);
        d3->green = (a3 * s3->green) + ((1 - a3) * b3->green);
        d3->blue = (a3 * s3->blue) + ((1 - a3) * b3->blue);
        d3->alpha = USHRT_MAX;

        d4->red = (a4 * s4->red) + ((1 - a4) * b4->red);
        d4->green = (a4 * s4->green) + ((1 - a4) * b4->green);
        d4->blue = (a4 * s4->blue) + ((1 - a4) * b4->blue);
        d4->alpha = USHRT_MAX;
    }
}

/*
 * register_blend_functions - Register all of your different versions
 *     of the blend kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_functions()
{
    add_blend_function(&blend, blend_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND_V KERNEL
 *****************************************************************************/

// Your different versions of the blend_v kernel go here
// (i.e. with vectorization, aka. SIMD).

// 256 bit vectors = 32 bytes
// One pixel contains 4 unsigned short values (16 bits each = 2 bytes)
// One pixel = 64 bits = 8 bytes
// One vector can hold 4 pixels
// I want to cast the shorts to floats (32 bits each = 4 bytes)
// Therefore, 2 pixels in one vector = 8 floats in one vector

/*
Function blend_pixel takes as argument three (pointers to) pixels: a source
pixel S , destination pixel D, and background pixel B. With α expressing the opacity of the source pixel as a ratio
(i.e. α = S.alpha / USHRT_MAX), blend_pixel computes, for each color c,
D.c = αS.c + (1 − α) * B.c
whilst making D fully opaque (i.e. D.alpha = USHRT_MAX).
*/
// TODO: Reuse variables instead of declaring new ones. 
char blend_v_descr[] = "blend_v: Current working version";
void blend_v(int dim, pixel *src, pixel *dst){
    // Vector with background color (two times using four floats for each)
    const __m256 bgv = _mm256_setr_ps(bgc.red, bgc.green, bgc.blue, 0.0, bgc.red, bgc.green, bgc.blue, 0.0);
    // Vector of all ones (floats) to be used to compute 1 - a
    const __m256 ones = _mm256_set1_ps(1.0); // [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1,0, 1.0 ]
    // Vector of 1.0 / USHRT_MAX to be used to compute a/USHRT_MAX
    const __m256 one_divided_by_max = _mm256_set1_ps(1.0 / USHRT_MAX);
    // Vector of all USHRT_MAX to be used to set alpha to USHRT_MAX at the very end of the loop.
    const __m256 maxes = _mm256_set1_ps(USHRT_MAX);

    // Loop unrolling of 16 pixels
    // Only use one loop instead of two because the picture is represented as a 1D array.
    // Each pixel can be processed independently of the others so we can use only one loop
    for (register int i = 0; i < dim * dim; i += 16) // Load 16 pixels at a time
    {
        /**************************** FIRST 4 PIXELS (0-4) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        __m256i pix4 = _mm256_load_si256((__m256i *)&src[i]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        // Convert to floats (4 bytes each)
        __m256 pix_float_even = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(pix4, _mm256_setzero_si256())); // Zero and second pixel
        __m256 pix_float_uneven = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(pix4, _mm256_setzero_si256())); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        __m256 alphas_even = _mm256_mul_ps((_mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7))), one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        __m256 alphas_uneven = _mm256_mul_ps((_mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7))), one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        // Round the result down to the nearest integer
        // Convert vectors of floats into vectors of ints
        __m256i result_int_even = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_blend_ps((_mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), (_mm256_mul_ps((_mm256_sub_ps(ones, alphas_even)), bgv)))), maxes, 0b10001000)));
        __m256i result_int_uneven = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_blend_ps((_mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), (_mm256_mul_ps((_mm256_sub_ps(ones, alphas_even)), bgv)))), maxes, 0b10001000)));

        // Store in dst [even0, uneven1, even2, uneven3] packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i], _mm256_packus_epi32(result_int_even, result_int_uneven));

        /**************************** NEXT 4 PIXELS (4-8) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        pix4 = _mm256_load_si256((__m256i *)&src[i + 4]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        result_int_even = _mm256_cvtps_epi32(result_even);
        result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i + 4], result_int);

        /**************************** NEXT 4 PIXELS (8-12) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        pix4 = _mm256_load_si256((__m256i *)&src[i + 8]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        result_int_even = _mm256_cvtps_epi32(result_even);
        result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i + 8], result_int);

        /**************************** NEXT 4 PIXELS (12-16) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        pix4 = _mm256_load_si256((__m256i *)&src[i + 12]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        result_int_even = _mm256_cvtps_epi32(result_even);
        result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i + 12], result_int);

        /**************************** DEBUGGING ****************************/
        /*
        // Print a short or int vector:
        unsigned short shorts[16];
        memcpy(shorts, &pix4, sizeof(pix4));
        printf("Loaded from src:\n");
        for (int z = 0; z < 8; z++)
        {
            printf("%*d:%*d, ", 2, z, 5, shorts[z]);
        }
        printf("\n");
        for (int z = 8; z < 16; z++)
        {
            printf("%*d:%*d, ", 2, z, 5, shorts[z]);
        }
        printf("\n");

        // Print a float vector
        float floatseven[8];
        memcpy(floatseven, &pix_float_even, sizeof(pix_float_even));
        printf("Contents of pix_float_even:\n");
        for (int z = 0; z < 8; z++)
        {
            printf("%*d:%f, ", 2, z, floatseven[z]);
        }
        printf("\n");

        */
    }
}

char blend_v_readable_descr[] = "blend_v: A version that is easier to read";
void blend_v_readable(int dim, pixel *src, pixel *dst)
{
    // Vector with background color (two times using four floats for each)
    const __m256 bgv = _mm256_setr_ps(bgc.red, bgc.green, bgc.blue, 0.0, bgc.red, bgc.green, bgc.blue, 0.0);
    // Vector of all ones (floats) to be used to compute 1 - a
    const __m256 ones = _mm256_set1_ps(1.0); // [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1,0, 1.0 ]
    // Vector of 1.0 / USHRT_MAX to be used to compute a/USHRT_MAX
    const __m256 one_divided_by_max = _mm256_set1_ps(1.0 / USHRT_MAX);
    // Vector of all USHRT_MAX to be used to set alpha to USHRT_MAX at the very end of the loop.
    const __m256 maxes = _mm256_set1_ps(USHRT_MAX);

    // Loop unrolling of 16 pixels
    // Only use one loop instead of two because the picture is represented as a 1D array.
    // Each pixel can be processed independently of the others so we can use only one loop
    for (register int i = 0; i < dim * dim; i += 16) // Load 16 pixels at a time
    {
        /**************************** FIRST 4 PIXELS (0-4) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        __m256i pix4 = _mm256_load_si256((__m256i *)&src[i]);
        /******************
        Contents of pix4:
        (red, green, blue, alpha)
        0 : 41810, 1 : 60127, 2 : 59456, 3 : 38956, 4 : 60275, 5 : 36251, 6 : 28336, 7 : 62482,
        8 : 35155, 9 : 30708, 10 : 60904, 11 : 25205, 12 : 5732, 13 : 38782, 14 : 25010, 15 : 61456,
        ******************/

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        __m256i pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        __m256i pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        __m256 pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        __m256 pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel
        /******************
        Contents of pix_float_even:
        (red, green, blue, alpha)
        0:41810.000000,  1:60127.000000,  2:59456.000000,  3:38956.000000,  4:35155.000000,  5:30708.000000,  6:60904.000000,  7:25205.000000,
        Contents of pix_float_uneven:
        0:60275.000000,  1:36251.000000,  2:28336.000000,  3:62482.000000,  4:5732.000000,  5:38782.000000,  6:25010.000000,  7:61456.000000,
        ******************/

        // Get alpha α from each pixel (at "index" 3 and 7)
        __m256 alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        __m256 alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        __m256 one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        __m256 one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        __m256 a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        __m256 a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        __m256 result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        __m256 result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        __m256i result_int_even = _mm256_cvtps_epi32(result_even);
        __m256i result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        __m256i result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i], result_int);

        /**************************** NEXT 4 PIXELS (4-8) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        pix4 = _mm256_load_si256((__m256i *)&src[i + 4]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        result_int_even = _mm256_cvtps_epi32(result_even);
        result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i + 4], result_int);

        /**************************** NEXT 4 PIXELS (8-12) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        pix4 = _mm256_load_si256((__m256i *)&src[i + 8]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        result_int_even = _mm256_cvtps_epi32(result_even);
        result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i + 8], result_int);

        /**************************** NEXT 4 PIXELS (12-16) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        pix4 = _mm256_load_si256((__m256i *)&src[i + 12]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        result_int_even = _mm256_cvtps_epi32(result_even);
        result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i + 12], result_int);

        /**************************** DEBUGGING ****************************/
        /*
        // Print a short or int vector:
        unsigned short shorts[16];
        memcpy(shorts, &pix4, sizeof(pix4));
        printf("Loaded from src:\n");
        for (int z = 0; z < 8; z++)
        {
            printf("%*d:%*d, ", 2, z, 5, shorts[z]);
        }
        printf("\n");
        for (int z = 8; z < 16; z++)
        {
            printf("%*d:%*d, ", 2, z, 5, shorts[z]);
        }
        printf("\n");

        // Print a float vector
        float floatseven[8];
        memcpy(floatseven, &pix_float_even, sizeof(pix_float_even));
        printf("Contents of pix_float_even:\n");
        for (int z = 0; z < 8; z++)
        {
            printf("%*d:%f, ", 2, z, floatseven[z]);
        }
        printf("\n");

        */
    }
}

char blend2_v_descr[] = "blend_v: Current working version";
void blend2_v(int dim, pixel *src, pixel *dst)
{
    // Vector with background color (two times using four floats for each)
    const __m256 bgv = _mm256_setr_ps(bgc.red, bgc.green, bgc.blue, 0.0, bgc.red, bgc.green, bgc.blue, 0.0);
    // Vector of all ones (floats) to be used to compute 1 - a
    const __m256 ones = _mm256_set1_ps(1.0); // [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1,0, 1.0 ]
    // Vector of 1.0 / USHRT_MAX to be used to compute a/USHRT_MAX
    const __m256 one_divided_by_max = _mm256_set1_ps(1.0 / USHRT_MAX);
    // Vector of all USHRT_MAX to be used to set alpha to USHRT_MAX at the very end of the loop.
    const __m256 maxes = _mm256_set1_ps(USHRT_MAX);

    // Loop unrolling of 16 pixels
    // Only use one loop instead of two because the picture is represented as a 1D array.
    // Each pixel can be processed independently of the others so we can use only one loop
    for (register int i = 0; i < dim * dim; i += 8) // Load 16 pixels at a time
    {
        /**************************** FIRST 4 PIXELS (0-4) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        __m256i pix4 = _mm256_load_si256((__m256i *)&src[i]);
        /******************
        Contents of pix4:
        (red, green, blue, alpha)
        0 : 41810, 1 : 60127, 2 : 59456, 3 : 38956, 4 : 60275, 5 : 36251, 6 : 28336, 7 : 62482,
        8 : 35155, 9 : 30708, 10 : 60904, 11 : 25205, 12 : 5732, 13 : 38782, 14 : 25010, 15 : 61456,
        ******************/

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        __m256i pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        __m256i pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        __m256 pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        __m256 pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel
        /******************
        Contents of pix_float_even:
        (red, green, blue, alpha)
        0:41810.000000,  1:60127.000000,  2:59456.000000,  3:38956.000000,  4:35155.000000,  5:30708.000000,  6:60904.000000,  7:25205.000000,
        Contents of pix_float_uneven:
        0:60275.000000,  1:36251.000000,  2:28336.000000,  3:62482.000000,  4:5732.000000,  5:38782.000000,  6:25010.000000,  7:61456.000000,
        ******************/

        // Get alpha α from each pixel (at "index" 3 and 7)
        __m256 alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        __m256 alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);             // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max);           // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        __m256 one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);      // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        __m256 one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven);    // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        __m256 a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        __m256 a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        __m256 result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        __m256 result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        __m256i result_int_even = _mm256_cvtps_epi32(result_even);
        __m256i result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        __m256i result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i], result_int);

        /**************************** NEXT 4 PIXELS (4-8) *****************************/

        // Vector with the first four pixels using shorts (2 bytes each)
        pix4 = _mm256_load_si256((__m256i *)&src[i+4]);

        // Convert to two vectors of ints (4 bytes each) https://www.officedaytime.com/simd512e/simdimg/unpack.php?f=punpckhwd
        pix_int_even = _mm256_unpacklo_epi16(pix4, _mm256_setzero_si256());   // Zero and second pixel
        pix_int_uneven = _mm256_unpackhi_epi16(pix4, _mm256_setzero_si256()); // First and third pixel

        // Convert to floats (4 bytes each)
        pix_float_even = _mm256_cvtepi32_ps(pix_int_even);     // Zero and second pixel
        pix_float_uneven = _mm256_cvtepi32_ps(pix_int_uneven); // First and third pixel

        // Get alpha α from each pixel (at "index" 3 and 7)
        alphas_even = _mm256_shuffle_ps(pix_float_even, pix_float_even, _MM_SHUFFLE(3, 3, 7, 7));       // [a0, a0, a0, a0, a2, a2, a2, a2]
        alphas_uneven = _mm256_shuffle_ps(pix_float_uneven, pix_float_uneven, _MM_SHUFFLE(3, 3, 7, 7)); // [a1, a1, a1, a1, a3, a3, a3, a3]
        // Compute α / USHRT_MAX (can also be written as a * 1/USHRT_MAX)
        alphas_even = _mm256_mul_ps(alphas_even, one_divided_by_max);     // [a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a0/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX, a2/USHRT_MAX]
        alphas_uneven = _mm256_mul_ps(alphas_uneven, one_divided_by_max); // [a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a1/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX, a3/USHRT_MAX]

        // Compute D.c = αS.c + (1 − α) * B.c
        // 1 - a
        one_minus_alpha_even = _mm256_sub_ps(ones, alphas_even);     // [1-a0, 1-a0, 1-a0, 1-a0, 1-a2, 1-a2, 1-a2, 1-a2]
        one_minus_alpha_uneven = _mm256_sub_ps(ones, alphas_uneven); // [1-a1, 1-a1, 1-a1, 1-a1, 1-a3, 1-a3, 1-a3, 1-a3]
        // (1 - a) * B.c
        a_times_b_c_even = _mm256_mul_ps(one_minus_alpha_even, bgv);
        a_times_b_c_uneven = _mm256_mul_ps(one_minus_alpha_uneven, bgv);
        // αS.c + (1 − α) * B.c
        result_even = _mm256_add_ps(_mm256_mul_ps(alphas_even, pix_float_even), a_times_b_c_even);
        result_uneven = _mm256_add_ps(_mm256_mul_ps(alphas_uneven, pix_float_uneven), a_times_b_c_uneven);

        // Set alphas at "index" [3, 7, 3, 7] to USHRT_MAX
        result_even = _mm256_blend_ps(result_even, maxes, 0b10001000); // blendv: https://www.officedaytime.com/simd512e/simdimg/si.php?f=blendvps
        result_uneven = _mm256_blend_ps(result_uneven, maxes, 0b10001000);

        // Round the result down to the nearest integer
        result_even = _mm256_floor_ps(result_even);
        result_uneven = _mm256_floor_ps(result_uneven);

        // Convert vectors of floats into vectors of ints
        result_int_even = _mm256_cvtps_epi32(result_even);
        result_int_uneven = _mm256_cvtps_epi32(result_uneven);

        // Store in dst [even0, uneven1, even2, uneven3]
        result_int = _mm256_packus_epi32(result_int_even, result_int_uneven); // packus: https://www.officedaytime.com/simd512e/simdimg/si.php?f=packssdw
        _mm256_store_si256((__m256i *)&dst[i+4], result_int);

        /**************************** DEBUGGING ****************************/
        /*
        // Print a short or int vector:
        unsigned short shorts[16];
        memcpy(shorts, &pix4, sizeof(pix4));
        printf("Loaded from src:\n");
        for (int z = 0; z < 8; z++)
        {
            printf("%*d:%*d, ", 2, z, 5, shorts[z]);
        }
        printf("\n");
        for (int z = 8; z < 16; z++)
        {
            printf("%*d:%*d, ", 2, z, 5, shorts[z]);
        }
        printf("\n");

        // Print a float vector
        float floatseven[8];
        memcpy(floatseven, &pix_float_even, sizeof(pix_float_even));
        printf("Contents of pix_float_even:\n");
        for (int z = 0; z < 8; z++)
        {
            printf("%*d:%f, ", 2, z, floatseven[z]);
        }
        printf("\n");

        */
    }
}

        /*
         * register_blend_v_functions - Register all of your different versions
         *     of the blend_v kernel with the driver by calling the
         *     add_blend_function() for each test function.
         */
        void register_blend_v_functions()
        {
            add_blend_v_function(&blend_v, blend_v_descr);
            add_blend_v_function(&blend2_v, blend2_v_descr);

            /* ... Register additional test functions here */
        }

        /******************************************************************************
         * SMOOTH KERNEL
         *****************************************************************************/

        // Your different versions of the smooth kernel go here

        /*
         * naive_smooth - The naive baseline version of smooth
         */
        char naive_smooth_descr[] = "naive_smooth: Naive baseline implementation";
        void naive_smooth(int dim, pixel *src, pixel *dst)
        {
            int i, j;

            for (i = 0; i < dim; i++)
                for (j = 0; j < dim; j++)
                    dst[RIDX(i, j, dim)] = avg(dim, i, j, src); // `avg` defined in smooth.c
        }

        char smooth_descr[] = "smooth: Current working version";
        void smooth(int dim, pixel *src, pixel *dst)
        {
            naive_smooth(dim, src, dst);
        }

        /*
         * register_smooth_functions - Register all of your different versions
         *     of the smooth kernel with the driver by calling the
         *     add_smooth_function() for each test function.
         */

        void register_smooth_functions()
        {
            add_smooth_function(&smooth, smooth_descr);
            /* ... Register additional test functions here */
        }
