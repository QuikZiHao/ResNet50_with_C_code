/*
 * @Descripttion: all function about front propagation method
 * @version: 2.0
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-01-26
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-04-30
 */
#ifndef _FRONTPROPAGATION_H
#define _FRONTPROPAGATION_H

/**
 * @name: Front_ReLU
 * @msg: ReLu function if value below than 0 equal to 0
 * @param {struct Matrix} *input               - (input)the feature which need to process ReLU
 * @param {struct Matrix} *output              - (output)the feature after process ReLU
 * @return {}
 */
void Front_ReLU (struct Matrix *input,struct Matrix *output);

/**
 * @name: Front_ReLU
 * @msg: ReLu function if value below than 0 equal to 0
 * @param {struct Matrix} **input               - (input)the feature batch which need to process ReLU
 * @param {struct Matrix} **output              - (output)the feature batch after process ReLU
 * @return {}
 */
void FrontBatch_ReLU (struct Matrix **input,struct Matrix **output);

/**
 * @name: Front_MaxPooLing
 * @msg: apply maxPooling process to the feature
 * @param {struct Matrix} *input      - (input)the feature which need to process max pooling
 * @param {struct Matrix} *output     - (output)the feature which after max pooling
 * @param {int} poolingSize            - (input)the size of the poolng kernel
 * @param {int} paddingAmt             - (input)the size need to be padding
 * @param {int} stride                 - (input)the stride(step) the kernel move per steps
 * @return {}
 */
void Front_MaxPooLing(struct Matrix *input,struct Matrix *output,int poolingSize,int paddingAmt,int stride);

/**
 * @name: FrontBatch_MaxPooLing
 * @msg: apply maxPooling process to the Batch feature 
 * @param {struct Matrix} **input      - (input)the Batch of feature which need to process max pooling
 * @param {struct Matrix} **output     - (output)the Batch of feature which after max pooling
 * @param {int} poolingSize            - (input)the size of the poolng kernel
 * @param {int} paddingAmt             - (input)the size need to be padding
 * @param {int} stride                 - (input)the stride(step) the kernel move per steps
 * @return {}
 */
void FrontBatch_MaxPooLing(struct Matrix **input,struct Matrix **output,int poolingSize,int paddingAmt,int stride);

/**
 * @name: FrontBatch_Convolution
 * @msg: Convolution the batch of feature with the kernel
 * @param {struct Matrix} **input             - (input)the image which need to process convolution
 * @param {struct Matrix} **kernel            - (input)the kernels which provided to convolution
 * @param {struct Matrix} **output            - (output)the Batch of struct which store the ans
 * @param {int} padingAmt                     - (input)the size need to be padding
 * @param {int} stride                        - (input)the stride(step) the kernel move per steps
 * @return {}
 */
void FrontBatch_Convolution(struct Matrix **input,struct Matrix **kernel,struct Matrix **output ,int paddingAmt,int stride);

/**
 * @name: Front_GlobalAverage
 * @msg: apply global average pooling process to the feature(after become feature = chX1X1)
 * @param {struct Matrix} *input      - (input)the feature which need to process global average pooling
 * @param {struct Matrix} *output     - (output)the feature which after global average pooling
 * @return {}
 */
void Front_GlobalAverage(struct Matrix *input,struct Matrix *output);

/**
 * @name: Front_GlobalAverage
 * @msg: apply global average pooling process to the batch of feature(after become feature = 1XchannelX1)
 * @param {struct Matrix} **input      - (input)the batch of feature which need to process global average pooling
 * @param {struct Matrix} **output     - (output)the batch of feature which after global average pooling
 * @return {}
 */
void FrontBatch_GlobalAverage(struct Matrix **input,struct Matrix **output);

/**
 * @name: Front_FullConnect
 * @msg: apply full connect to the fecture (1XrowX1) with weight (1XpredictsizeXrow) and return (1XpredictsizeX1)
 * @param {struct Matrix} **input      - (input)the feature which need to fullconnect
 * @param {struct Matrix} *weight      - (input)the weight of the fc layer
 * @param {struct Matrix} **output     - (output)the feature after fullconncect
 * @param {struct Matrix} *bias        - (input)the bias of the fc layer 
 * @return {}
 */
void Front_FullConnect(struct Matrix *input,struct Matrix *weight,struct Matrix *output,struct Matrix *bias);

/**
 * @name: FrontBatch_FullConnect
 * @msg: apply full connect to the batch of fecture (1XrowX1) with weight (1XpredictsizeXrow) and return (1XpredictsizeX1)
 * @param {struct Matrix} **input      - (input)the batch of feature which need to fullconnect
 * @param {struct Matrix} *weight      - (input)the weight of the fc layer
 * @param {struct Matrix} **output     - (output)the batch of feature after fullconncect
 * @param {struct Matrix} *bias        - (input)the bias of the fc layer 
 * @return {}
 */
void FrontBatch_FUllConnct(struct Matrix **input,struct Matrix *weight,struct Matrix **output,struct Matrix *bias);

/**
 * @name: Front_Softmax
 * @msg: activation function 
 * @param {struct Matrix} *input              - (input)the feature need to be activated
 * @param {int} maximun                       - (input) index of the maximun term
 * @param {struct Matrix} *output             - (output)the feature after activated
 * @return {}
 */
void Front_Softmax(struct Matrix *input,int maximun,struct Matrix *output);

/**
 * @name: FrontBatch_Softmax
 * @msg: activation function for batch
 * @param {struct Matrix} **input              - (input)the batch of feature need to be activated
 * @param {int} *maximun                       - (input) the maximun of each logits
 * @param {struct Matrix} **output             - (output)the batch of feature after activated
 * @return {}
 */
void FrontBatch_Softmax(struct Matrix **input,int *maximun,struct Matrix **output);

/**
 * @name: Front_Predict
 * @msg: return the prediction of the front propagation
 * @param {struct Matrix} *input                - (input)the values list after softmax
 * @return {int} the index of the prediction
 */
int Front_Predict(struct Matrix *input);

/**
 * @name: FrontBatch_Predict
 * @msg: return the prediction lisy of the front propagation
 * @param {struct Matrix} **input                - (input)the values list after softmax
 * @param {int} *output                          - (output)the predict ans list
 * @return {}
 */
void FrontBatch_Predict(struct Matrix **input, int *output);

/**
 * @name: Res_Sum
 * @msg: residual block with Sum
 * @param {struct Matrix} *input          - (input)the term after bottleneck
 * @param {struct Matrix} *lastInput      - (input)the term before bottleneck
 * @param {struct Matrix} *output         - (output)the ans of residual
 * @return {}
 */
void Res_Sum(struct Matrix *input , struct Matrix *lastInput, struct Matrix *output);

/**
 * @name: Res_Sum
 * @msg: residual block with Sum
 * @param {struct Matrix} **input          - (input)the batch of term after bottleneck
 * @param {struct Matrix} **lastInput      - (input)the batch of term before bottleneck
 * @param {struct Matrix} **output         - (output)the batch of ans of residual
 * @return {}
 */
void ResBatch_Sum(struct Matrix **input , struct Matrix **lastInput, struct Matrix **output);

/**
 * @name: Front_Accurancy
 * @msg: calculate the error of the iterate
 * @param {int} *input                    - (input)the list of each predict values
 * @param {int} *testCaseLabel            - (input)the list of ans of each picture
 * @param {int} index                     - (input)the firstindex of the list
 * @return {float} the value of the accurancy(error) 
 */
float FrontBatch_Accurancy(int *input,int *testCaseLabel,int index);

/**
 * @name: Front_LossFunction
 * @msg: calculate the loss function of the iterate
 * @param {struct Matrix} **input         - (input)batch of the feature after softmax
 * @param {int} *testCaseLabel            - (input)the list of ans of each picture
 * @param {int} index                     - (input)the firstindex of the list
 * @return {float} the value of the costfunction(error) 
 */
float FrontBatch_LossFunction(struct Matrix **input,int *testCaseLabel,int index);
#endif