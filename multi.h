/*
 * @Descripttion: para calculate(multithread)
 * @version:1.0
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-03-25
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-04-29
 * @massage: 
 */
#ifndef _MULTI_H
#define _MULTI_H

#define THREADSIZE 8

struct ConvolutionArgs
{
    struct Matrix *input;
    struct Matrix **kernel;
    struct Matrix *output;
    int paddingAmt;
    int stride;
};
/*
struct Args
{
    struct Matrix *input;
    struct Matrix *output;
};
*/

struct MaxArgs
{
    struct Matrix *input;
    struct Matrix *output;
    int poolingSize;
    int paddingAmt;
    int stride;
};

pthread_t th[THREADSIZE]; // thread = 16
struct ConvolutionArgs CONV_Args[THREADSIZE]; //variable for convolution
//struct Args matrixArgs[THREADSIZE]; //variable for ReLU and to zero
struct MaxArgs Max_Args[THREADSIZE]; //variable for maxPooling

/**
 * @name: Thread_Convolution
 * @msg: convolution in thread 
 * @param {struct ConvolutionArgs} args      - (input)all parameter of convolution
 * @return {}
 */
void* Thread_Convolution(void *args);

/**
 * @name: Multi_Convolution
 * @msg: Convolution the batch of feature with the kernel with multiple thread
 * @param {struct Matrix} **input             - (input)the image which need to process convolution
 * @param {struct Matrix} **kernel            - (input)the kernels which provided to convolution
 * @param {struct Matrix} **output            - (output)the Batch of struct which store the ans
 * @param {int} padingAmt                     - (input)the size need to be padding
 * @param {int} stride                        - (input)the stride(step) the kernel move per steps
 * @return {*}
 */
void Multi_Convolution(struct Matrix **input,struct Matrix **kernel,struct Matrix **output ,int paddingAmt,int stride);

/**
 * @name: Thread_Convolution
 * @msg: convolution in thread 
 * @param {struct Args} args      - (input)all parameter of ReLU
 * @return {}
 */
//void* Thread_ReLU(void *args);

/**
 * @name: Multi_ReLU
 * @msg: ReLu function if value below than 0 equal to 0 with multiple thread
 * @param {struct Matrix} **input               - (input)the feature batch which need to process ReLU
 * @param {struct Matrix} **output              - (output)the feature batch after process ReLU
 * @return {*}
 */
//void Multi_ReLU (struct Matrix **input,struct Matrix **output);

/**
 * @name: Thread_Convolution
 * @msg: maxpooling in thread 
 * @param {struct MaxArgs} args      - (input)all parameter of ReLU
 * @return {}
 */
void* Thread_MaxPooLing(void *args);

/**
 * @name: Multi_MaxPooLing
 * @msg: apply maxPooling process to the Batch feature with multithread
 * @param {struct Matrix} **input      - (input)the Batch of feature which need to process max pooling
 * @param {struct Matrix} **output     - (output)the Batch of feature which after max pooling
 * @param {int} poolingSize            - (input)the size of the poolng kernel
 * @param {int} paddingAmt             - (input)the size need to be padding
 * @param {int} stride                 - (input)the stride(step) the kernel move per steps
 * @return {*}
 */
void Multi_MaxPooLing(struct Matrix **input,struct Matrix **output,int poolingSize,int paddingAmt,int stride);


/**
 * @name: Thread_ToZero
 * @msg: let gradient be zero in thread 
 * @param {struct MaxArgs} args      - (input)all parameter of ReLU
 * @return {}
 */
//void* Thread_ToZero(void *args);

/**
 * @name: Multi_ReLU
 * @msg: clear the previous gradient to zero with multiple thread
 * @param {struct Matrix} **input               - (input)the gradient batch which need to process to zero
 * @return {*}
 */
//void Multi_ToZero (struct Matrix **input);

/**
 * @name: Thread_Convolution_Variable
 * @msg: calculate the gradient convolution layer in multiple thread 
 * @param {struct MaxArgs} args      - (input)all parameter of gradient of conv
 * @return {}
 */
void* Thread_Convolution_Variable(void *args);

/**
 * @name: Multi_Convolution
 * @msg: calculate the Convoulution layer gradient and gradient descent the kernel in multiple thread
 * @param {struct Matrix} **lastTermGradient   - (input)the batch of gradient after convolution layer
 * @param {struct Matrix} **kernel             - (output)the convolution kernel in this layer
 * @param {struct Matrix} **gradient           - (output)this layer gradient
 * @param {struct Matrix} **variable           - (input)the batch of matrix before convolution
 * @param {struct Matrix} **gradientKernel     - (input)this layer kernel gradient
 * @param {int} stride                         - (input)the stride(step) the kernel move per steps
 * @param {int} padingAmt                      - (input)the size need to be padding
 * @return {} 
 */
void Multi_GradientConvolution(struct Matrix **lastTermGradient ,struct Matrix **kernel ,struct Matrix **gradient,
                           struct Matrix **variable ,struct Matrix **gradientKernel, int stride, int paddingAmt);
#endif