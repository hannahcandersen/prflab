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
void rotate(int dim, pixel *restrict src, pixel *restrict dst) // restrict keyword: compiler can assume that `src` and `dst` do not overlap
{
    register int i, j;
    register int dim_minus_1 = dim - 1;

    for (i = 0; i < dim; i += 4)
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
        rotate(dim, src, dst);
        return;
    }

    pthread_t threads[4];
    thread_args thread_data[4];

    int rows_per_thread = dim / 4;

    for (int t = 0; t < 4; t++)
    {
        thread_data[t].dim = dim;
        thread_data[t].src = src;
        thread_data[t].dst = dst;
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t == 3) ? dim : (t + 1) * rows_per_thread;

        pthread_create(&threads[t], NULL, rotate_thread, (void *)&thread_data[t]);
    }

    for (int t = 0; t < 4; t++)
    {
        pthread_join(threads[t], NULL);
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

char blend_descr[] = "blend: Current working version";
void blend(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
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

char blend_v_descr[] = "blend_v: Current working version";
void blend_v(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
}

/*
 * register_blend_v_functions - Register all of your different versions
 *     of the blend_v kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_v_functions()
{
    add_blend_v_function(&blend_v, blend_v_descr);
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
