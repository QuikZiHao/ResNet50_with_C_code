/*
 * @Descripttion: all function about back propagation
 * @version: 2.0
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-02-03
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-04-30
 */
#ifndef _BACKPROPAGATION_H
#define _BACKPROPAGATION_H

#define LEARNINGRATE 0.1 //0.1/batch size

/**
 * @name: Back_BatchNorm_Descent
 * @msg: gradeint list clear to zero
 * @param {float} *gradient       - (output)the gradient need to be zero
 * @param {int} size              - (input)the size of the gradient list
 * @return {} 
 */
void Back_ToZero(float *gradient,int size);

/**
 * @name: Back_Descent
 * @msg: gradeint descent the kernel when back propagation
 * @param {struct Matrix} *kernel         - (output)the kernel need to be gradient descent
 * @param {struct Matrix} *gradient       - (input)the gradient of the layer
 * @return {} 
 */
void Back_Descent(struct Matrix *kernel,struct Matrix *gradient);

/**
 * @name: Back_BatchNorm_Descent
 * @msg: gradeint descent the BN beta and gamma when back propagation
 * @param {struct BN} *weight         - (output)the kernel need to be gradient descent
 * @param {float} *gradientBeta       - (input)the gradient Beta of the BatchNorm
 * @param {float} *gradientGamma      - (input)the gradient Gamma of the BatchNorm
 * @return {} 
 */
void Back_BatchNorm_Descent(struct BN *weight,float *gradientBeta,float *gradientGamma);

/**
 * @name: Back_CostFunction
 * @msg: calculate the gradeint of cost function
 * @param {struct Matrix} *input          - (input)the matrix of soft max values
 * @param {int} testCaseLabel             - (input)the ans of the picture picture
 * @param {struct Matrix} *gradient       - (output)the gradient of the layer
 * @return {} 
 */
void Back_CostFunction(struct Matrix *input,int testCaseLabel,struct Matrix *gradient);

/**
 * @name: Gradient_CostFunction
 * @msg: calculate the gradeint of the cost function with batch 
 * @param {struct Matrix} **input          - (input)the batch matrix of soft max values
 * @param {int} *testCaseLabel             - (input)the batch ans of the picture picture
 * @param {struct Matrix} **gradient       - (output)the gradient of the layer with batch
 * @return {} 
 */
void Gradient_CostFunction(struct Matrix **input,int *testCaseLabel , int index , struct Matrix **gradient);

/**
 * @name: Back_FullConnct_Bias
 * @msg: calculate the gradeint of the FC Bias
 * @param {struct Matrix} *lastTermGradeint   - (input)the gradient soft max layer
 * @param {struct Matrix} *gradient           - (output)the gradient of the bias
 * @return {} 
 */
void Back_FullConnect_Bias(struct Matrix *lastTermGradeint,struct Matrix *gradient);

/**
 * @name: Gradient_FullConnct_Bias
 * @msg: calculate the gradeint of the batch of FC Bias
 * @param {struct Matrix} **lastTermGradeint   - (input)the batch of gradient soft max layer
 * @param {struct Matrix} *bias               - (input)the bias of the fc layer 
 * @param {struct Matrix} *gradient           - (output)the gradient of the bias
 * @return {} 
 */
void Gradient_FullConnect_Bias(struct Matrix **lastTermGradeint,struct Matrix *bias,struct Matrix *gradient);

/**
 * @name: Back_FullConnct_Weight
 * @msg: calculate the gradient of the FC Weight
 * @param {struct Matrix} *lastTermGradeint   - (input)the gradient soft max layer
 * @param {struct Matrix} *variable           - (input)the matrix of FC layer
 * @param {struct Matrix} *gradient           - (output)the gradient of the FC weight
 * @return {} 
 */
void Back_FullConnect_Weight(struct Matrix *lastTermGradient ,struct Matrix *variable, struct Matrix *gradient);

/**
 * @name: Gradient_FullConnct_Weight
 * @msg: calculate the batch of gradient FC Weight
 * @param {struct Matrix} **lastTermGradient   - (input)the batch of gradient soft max layer
 * @param {struct Matrix} **variable           - (input)the batch of matrix FC layer
 * @param {struct Matrix} *weight              - (output)the weight of the FC layer
 * @param {struct Matrix} *gradient            - (input)the gradient of the FC weight
 * @return {} 
 */
void Gradient_FullConnect_Weight(struct Matrix **lastTermGradient ,struct Matrix **variable,struct Matrix *weight, struct Matrix *gradient);

/**
 * @name: Back_FullConnct_Variable
 * @msg: calculate this layer gradient
 * @param {struct Matrix} *lastTermGradient   - (input)the gradient soft max layer
 * @param {struct Matrix} *weight             - (input)the weight of FC layer
 * @param {struct Matrix} *gradient           - (output)this layer gradient
 * @return {} 
 */
void Back_FullConnect_Variable(struct Matrix *lastTermGradient ,struct Matrix *weight, struct Matrix *gradient);

/**
 * @name: Gradient_FullConnct_Variable
 * @msg: calculate the batch of this layer gradient 
 * @param {struct Matrix} **lastTermGradient   - (input)the batch of gradient soft max layer
 * @param {struct Matrix} *weight              - (input)the weight of FC layer
 * @param {struct Matrix} **gradient           - (output)the batch of this layer gradient
 * @return {} 
 */
void Gradient_FullConnect_Variable(struct Matrix **lastTermGradient ,struct Matrix *weight, struct Matrix **gradient);

/**
 * @name: Gradient_FullConnct
 * @msg: gradient descent of this layer and calculate this layer gradient 
 * @param {struct Matrix} **lastTermGradient   - (input)the batch of gradient soft max layer
 * @param {struct Matrix} *bias                - (input)the bias of the fc layer 
 * @param {struct Matrix} *gradientBias        - (output)the gradient of the bias
 * @param {struct Matrix} *weight              - (output)the weight of FC layer
 * @param {struct Matrix} **variable           - (input)the batch of matrix FC layer
 * @param {struct Matrix} *gradientWeight      - (input)the gradient of the FC weight
 * @param {struct Matrix} **gradient           - (output)the batch of this layer gradient
 * @return {} 
 */
void Gradient_FullConnect(struct Matrix **lastTermGradient ,struct Matrix *bias ,struct Matrix *gradientBias
                        ,struct Matrix *weight , struct Matrix **variable , struct Matrix *gradientWeight
                        ,struct Matrix **gradient);

/**
 * @name: Back_GlobalAverage
 * @msg: calculate this layer gradient 
 * @param {struct Matrix} *lastTermGradient    - (input)gradient FC layer
 * @param {struct Matrix} *variable            - (input)matrix global average layer
 * @param {struct Matrix} *gradient            - (output)this layer gradient
 * @return {} 
 */
void Back_GlobalAverage(struct Matrix *lastTermGradient ,struct Matrix *variable, struct Matrix *gradient);

/**
 * @name: Gradient_GlobalAverage
 * @msg: calculate the batch of this layer gradient 
 * @param {struct Matrix} **lastTermGradient    - (input)the batch of gradient FC layer
 * @param {struct Matrix} **variable            - (input)the batch of matrix global average layer
 * @param {struct Matrix} **gradient            - (output)the batch of this layer gradient
 * @return {} 
 */
void Gradient_GlobalAverage(struct Matrix **lastTermGradient ,struct Matrix **variable, struct Matrix **gradient);

/**
 * @name: Back_MaxPooling
 * @msg: calculate this layer gradient 
 * @param {struct Matrix} *lastTermGradient    - (input)gradient after maxpooling layer
 * @param {struct Matrix} *variable            - (input)matrix max pooling layer
 * @param {struct Matrix} *output              - (input)matrix after max pooling layer
 * @param {int} poolingSize                    - (input)the size of the poolng kernel
 * @param {int} paddingAmt                     - (input)the size need to be padding
 * @param {int} stride                         - (input)the stride(step) the kernel move per steps
 * @param {struct Matrix} *gradient            - (output)this layer gradient
 * @return {} 
 */
void Back_MaxPooling(struct Matrix *lastTermGradient ,struct Matrix *variable ,struct Matrix *output,int poolingSize,int paddingAmt,int stride,struct Matrix *gradient);

/**
 * @name: Gradient_MaxPooling
 * @msg: calculate the batch of this layer gradient 
 * @param {struct Matrix} **lastTermGradient    - (input)the batch of gradient after maxpooling layer
 * @param {struct Matrix} **variable            - (input)the batch of matrix max pooling layer
 * @param {struct Matrix} **output              - (input)the batch of matrix after max pooling layer
 * @param {int} poolingSize                     - (input)the size of the poolng kernel
 * @param {int} paddingAmt                      - (input)the size need to be padding
 * @param {int} stride                          - (input)the stride(step) the kernel move per steps
 * @param {struct Matrix} **gradient            - (output)the batch of this layer gradient
 * @return {} 
 */
void Gradient_MaxPooling(struct Matrix **lastTermGradient ,struct Matrix **variable ,struct Matrix **output,int poolingSize,int paddingAmt,int stride,struct Matrix **gradient);

/**
 * @name: Back_ReLU
 * @msg: calculate ReLU layer gradient 
 * @param {struct Matrix} *lastTermGradient    - (input)gradient after ReLU layer
 * @param {struct Matrix} *variable            - (input)matrix ReLU layer
 * @param {struct Matrix} *gradient            - (output)this layer gradient
 * @return {} 
 */
void Back_ReLU(struct Matrix *lastTermGradient , struct Matrix *variable, struct Matrix *gradient);

/**
 * @name: Gradient_ReLU
 * @msg: calculate the batch of ReLU layer gradient 
 * @param {struct Matrix} **lastTermGradient    - (input)the batch of gradient after ReLU layer
 * @param {struct Matrix} **variable            - (input)the batch of matrix ReLU layer
 * @param {struct Matrix} **gradient            - (output)the batch of this layer gradient
 * @return {} 
 */
void Gradient_ReLU(struct Matrix **lastTermGradient , struct Matrix **variable, struct Matrix **gradient);

/**
 * @name: Gradient_BatchNorm_Variable
 * @msg: calculate the batch of batch normalize layer gradient 
 * @param {struct Matrix} **lastTermGradient    - (input)the batch of gradient after batchnorm layer
 * @param {struct Matrix} **variable            - (input)the batch of matrix before batchnorm
 * @param {struct BN} *coeff                    - (input)the coeff(mean,variance,beta,gama)
 * @param {struct Matrix} **gradient            - (output)the batch of this layer gradient
 * @return {} 
 */
void Gradient_BatchNorm_Variable(struct Matrix **lastTermGradient , struct Matrix **variable,struct BN *coeff,struct Matrix **gradient);

/**
 * @name: Gradient_BatchNorm_Weight
 * @msg: calculate batch normalize layer beta and gamma gradient and process gradient descent to the beta and gamma
 * @param {struct Matrix} **lastTermGradient    - (input)gradient after batchnorm layer
 * @param {struct BN} *coeff                    - (output)the coeff(mean,variance,beta,gama)
 * @param {struct Matrix} **variable            - (input)matrix after batchnorm
 * @param {float} *gradientBeta                 - (input)this layer beta gradient total batch
 * @param {float} *gradientGamma                - (input)this layer gamma gradient total batch
 * @return {} 
 */
void Gradient_BatchNorm_Weight(struct Matrix **lastTermGradient ,struct BN *coeff,
                          struct Matrix **variable,float *gradientBeta,float *gradientGamma);

/**
 * @name: Gradient_BatchNorm
 * @msg: calculate the batch of batch normalize layer gradient 
 * @param {struct Matrix} **lastTermGradient    - (input)the batch of gradient after batchnorm layer
 * @param {struct Matrix} **inputVariable       - (input)the batch of matrix before batchnorm
 * @param {struct BN} *coeff                    - (input)the coeff(mean,variance,beta,gama)
 * @param {struct Matrix} **gradient            - (output)the batch of this layer gradient
 * @param {struct Matrix} **outputVariable      - (input)the batch of matrix after batchnorm
 * @param {float} *gradientBeta                 - (input)this layer beta gradient total batch
 * @param {float} *gradientGamma                - (input)this layer gamma gradient total batch
 * @return {} 
 */
void Gradient_BatchNorm(struct Matrix **lastTermGradient , struct Matrix **inputVariable,struct BN *coeff
                        , struct Matrix **gradient ,struct Matrix **outputVariable,float *gradientBeta,float *gradientGamma );

/**
 * @name: Back_Convolution_Variable
 * @msg: calculate Convoulution layer gradient 
 * @param {struct Matrix} *lastTermGradient    - (input)gradient after convolution layer
 * @param {struct Matrix} **kernel             - (input)the convolution kernel in this layer
 * @param {struct Matrix} *gradient            - (output)this layer gradient
 * @param {int} stride                         - (input)the stride(step) the kernel move per steps
 * @param {int} padingAmt                      - (input)the size need to be padding
 * @return {} 
 */
void Back_Convolution_Variable(struct Matrix *lastTermGradient ,struct Matrix **kernel ,struct Matrix *gradient,
                            int stride, int paddingAmt);

/**
 * @name: Gradient_BatchNorm_Variable
 * @msg: calculate the batch of Convoulution layer gradient 
 * @param {struct Matrix} **lastTermGradient   - (input)the batch of gradient after convolution layer
 * @param {struct Matrix} **kernel             - (input)the convolution kernel in this layer
 * @param {float} **gradient                   - (output)the batch of this layer gradient
 * @param {int} stride                         - (input)the stride(step) the kernel move per steps
 * @param {int} padingAmt                      - (input)the size need to be padding
 * @return {} 
 */
void Gradient_Convolution_Variable(struct Matrix **lastTermGradient ,struct Matrix **kernel ,
                                    struct Matrix **gradient,int stride, int paddingAmt);


/**
 * @name: Back_Convolution_Kernel
 * @msg: calculate Convoulution layer kernel gradient 
 * @param {struct Matrix} *lastTermGradient    - (input)gradient after convolution layer
 * @param {struct Matrix} **gradient           - (output)this layer kernel gradient
 * @param {struct Matrix} *variable            - (input)matrix before convolution
 * @param {int} stride                         - (input)the stride(step) the kernel move per steps
 * @param {int} padingAmt                      - (input)the size need to be padding
 * @return {} 
 */
void Back_Convolution_Kernel(struct Matrix *lastTermGradient ,struct Matrix **gradient,struct Matrix *variable 
                            ,int stride, int paddingAmt);

/**
 * @name: Gradient_Convolution_Kernel
 * @msg: calculate the batch of Convoulution layer kernel gradient 
 * @param {struct Matrix} **lastTermGradient   - (input)the batch of gradient after convolution layer
 * @param {struct Matrix} **gradient           - (output)this layer kernel gradient
 * @param {struct Matrix} **variable           - (input)the batch of matrix before convolution
 * @param {int} stride                         - (input)the stride(step) the kernel move per steps
 * @param {int} padingAmt                      - (input)the size need to be padding
 * @return {} 
 */
void Gradient_Convolution_Kernel(struct Matrix **lastTermGradient ,struct Matrix **gradient,struct Matrix **variable ,int stride, int paddingAmt);


/**
 * @name: Gradient_Convolution
 * @msg: calculate the Convoulution layer gradient and gradient descent the kernel
 * @param {struct Matrix} **lastTermGradient   - (input)the batch of gradient after convolution layer
 * @param {struct Matrix} **kernel             - (output)the convolution kernel in this layer
 * @param {struct Matrix} **gradient           - (output)this layer gradient
 * @param {struct Matrix} **variable           - (input)the batch of matrix before convolution
 * @param {struct Matrix} **gradientKernel     - (input)this layer kernel gradient
 * @param {int} stride                         - (input)the stride(step) the kernel move per steps
 * @param {int} padingAmt                      - (input)the size need to be padding
 * @return {} 
 */
void Gradient_Convolution(struct Matrix **lastTermGradient ,struct Matrix **kernel ,struct Matrix **gradient,
                           struct Matrix **variable ,struct Matrix **gradientKernel, int stride, int paddingAmt);
#endif