/*
 * @Descripttion: all function about batch normalization
 * @version: 2.0
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-01-28
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-04-28
 */
#ifndef _BATCHNORM_H
#define _BATCHNORM_H

#define BATCH 32 //use this to select the size of each batch
#define EPSILON 0.1  // the minimun value of float
#define MOMENTUM 0.1 // the moving average of sample mean and variance in a mini-batch for training

struct BN
{
    int channelSize;

    float *beta;
    float *gamma;
    float *mean;
    float *variance;
    float *runningMean;  //use for test case
    float *runningVar; //use for test case
};

/**
 * @name: BN_Init
 * @msg: initial the structure to store the coeff of batch norm
 * @param {int} channel               - (input)the size of the channel in this layer
 * @return {struct BN*} *input        - (output)the struct bn after initial
 */
struct BN *BN_Init(int channel);

/**
 * @name: BN_GetCoeff
 * @msg: calculate the mean and variance of the batch
 * @param {struct Matrix} **input         - (input)the list of the feature
 * @param {struct BN} *coeff              - (output)the coeff(mean,variance,beta,gama)
 * @return {*}
 */
void *BN_GetCoeff(struct Matrix **input , struct BN *coeff);

/**
 * @name: BN_BatchNorm
 * @msg: let the feature proceed batch normalization function
 * @param {struct Matrix} *input          - (input)the list of the feature
 * @param {struct Matrix} *output         - (output)the list of the feature
 * @param {struct BN} *coeff              - (input)the coeff(mean,variance,beta,gama)
 * @return {} 
 */
void BN_BatchNorm(struct Matrix *input ,struct Matrix *output ,struct BN *coeff);

/**
 * @name: BNList_BatchNorm
 * @msg: let the feature list proceed batch normalization function
 * @param {struct Matrix}**input                      - (input)the feature batch need to be normalize
 * @param {struct Matrix}**output                     - (input)the feature batch after normalize
 * @param {struct BN} *coeff                          - (input)the coeff(mean and variance , gamma , beta)
 * @return {*}
 */
void **BNBatch_BatchNorm(struct Matrix **input ,struct Matrix **output, struct BN *coeff );



#endif