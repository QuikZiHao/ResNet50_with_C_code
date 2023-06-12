/*
 * @Descripttion: input the learning/test case to the program
 * @version:2.0
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-01-24 
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-04-30
 */
#ifndef _LEARNING_H
#define _LEARNING_H

/**
 * @name: FrontPropagation
 * @msg: frontpropagation and return the error
 * @param {int} index             - (input)the first number of the trainningcase index
 * @return {float}the error of the front propagation
 */
float FrontPropagation(int index);

/**
 * @name: BackPropagation
 * @msg: backpropagation and descent all the weight in the network
 * @param {int} index             - (input)the first number of the trainningcase index
 * @return {}
 */
void BackPropagation(int index);

/**
 * @name: Record_Kernel
 * @msg: record the kernel which learnt
 * @param {char} *fileName        - (input)the filelocation which storage the kernel info
 * @return {}
 */
void Record_Kernel(char *fileName);


/**
 * @name: Write_Kernel
 * @msg: write the kernel which learnt by format all value with space
 * @param {FILE} *kernelFile                  - (output)the file to edit
 * @param {struct Matrix} **kernel            - (input)the kernel which need to be record
 * @param {int} kernelAmt                     - (input)the amount of the kernel
 * @return {}
 */
void Write_Kernel(FILE *kernelFile,struct Matrix **kernel,int kernelAmt);


/**
 * @name: Write_Matrix
 * @msg: write the Matrix which learnt by format 
 *          weightName-channel-row-column\n
 *          all value with space
 * @param {FILE} *kernelFile                  - (output)the file to edit
 * @param {struct Matrix} *weight             - (input)the weight which need to be record
 * @param {char} *weightName                  - (input)the Name of the weight
 * @return {}
 */
void Write_Matrix(FILE *kernelFile,struct Matrix *weight,struct Matrix *bias);


/**
 * @name: Write_BN
 * @msg: write the BN which learnt by format 
 *          coeffName-channel\n
 *          beta[i] gamma[i] mean[i] variance[i]\n byperchannel
 * @param {FILE} *kernelFile                  - (output)the file to edit
 * @param {struct BN} *coeff                  - (input)the BN which need to be record
 * @param {char} *coeffName                   - (input)the Name of the coeff
 * @return {}
 */
void Write_BN(FILE *kernelFile, struct BN *coeff);
#endif