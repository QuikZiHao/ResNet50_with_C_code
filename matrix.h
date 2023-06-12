/*
 * @Descripttion: all function about the feature Matrix
 * @version:
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-01-25
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-03-12
 */

#ifndef _MATRIX_H
#define _MATRIX_H

struct Matrix
{
    int channelSize;
    int rowSize;
    int columnSize;

    float ***feature;

};

/**
 * @name: Matrix_Init
 * @msg:  initial the feature Matrix size with zero or random num (0-255)
 * @param {{void} *func                    - (input)select the initial mode
 * @param {int} channel                    - (input)the size of the channel
 * @param {int} row                        - (input)the size of the row
 * @param {int} column                     - (input)the size of the column
 * @return {struct Matrix*}the feature after init
 */
struct Matrix *Matrix_Init(struct Matrix *(*func)(int ,int,int),int channel ,int row,int column);

/**
 * @name: Zero
 * @msg: initial the struct Matrix with all feature equal to 0 
 * @param {int} channel                    - (input)the size of the channel
 * @param {int} row                        - (input)the size of the row
 * @param {int} column                     - (input)the size of the column
 * @return {*}
 */
struct Matrix *Zero(int channel , int row,int column);

/**
 * @name: Random
 * @msg: initial the struct Matrix with all feature random (0-1) 
 * @param {int} channel                    - (input)the size of the channel
 * @param {int} row                        - (input)the size of the row
 * @param {int} column                     - (input)the size of the column
 * @return {*}
 */
struct Matrix *Random(int channel , int row,int column);

/**
 * @name: Matrix_Print
 * @msg: output the feature Matrix in properly format
 * @param {struct Matrix} *input            - (output)the feature Matrix
 * @return {*}
 */
void Matrix_Print( struct Matrix *input );

/**
 * @name: Matrix_Show
 * @msg: output the feature Matrix which can show at opencv
 * @param {struct Matrix} *input            - (output)the feature Matrix
 * @return {*}
 */
void Matrix_Show( struct Matrix *input );

/**
 * @name: Matrix_Check
 * @msg: output the size of the feature matrix
 * @param {struct Matrix} *input            - (output)the feature Matrix
 * @return {*}
 */
void Matrix_Check(struct Matrix *input);

/**
 * @name: Matrix_Convolution
 * @msg: 3D image and 3D kernel convolution (dot product) p/s:channel of input should equal channel of kernel
 * @param {struct Matrix} *input             - (input)the image which need to process convolution
 * @param {struct Matrix} **kernel            - (input)the kernels which provided to convolution
 * @param {struct Matrix} *output            - (output)the struct which store the ans
 * @param {int} padingAmt                    - (input)the size need to be padding
 * @param {int} stride                       - (input)the stride(step) the kernel move per steps
 * @return {*}
 */
void Matrix_Convolution(struct Matrix *input,struct Matrix **kernel,struct Matrix *output ,int paddingAmt,int stride);

/**
 * @name: Matrix_Resize
 * @msg: resize the matrix size
 * @param {struct Matrix} *input             - (input)the image which need to process convolution
 * @param {int} channel                      - (input)the size of the channel
 * @param {int} row                        - (input)the size of the row
 * @param {int} column                     - (input)the size of the column
 * @return {*}
 */
void Matrix_Resize(struct Matrix *input,int channel,int row , int column);

/**
 * @name: Matrix_Form
 * @msg: resize the matrix size
 * @param {int} channel                      - (input)the size of the channel
 * @param {int} row                        - (input)the size of the row
 * @param {int} column                     - (input)the size of the column
 * @return {float ***}the feature matrix which after form
 */
float ***Matrix_Form(int channel,int row , int column);

/**
 * @name: Matrix_Multiply
 * @msg: multiply the Matrix with multiply term and return the output
 * @param {struct Matrix} *input              - (input)the matrix which need to process multiply
 * @param {struct Matrix} *mulTerm            - (input)the mulTerm which being multiply
 * @param {struct Matrix} *output             - (output)the matrix which store the solution
 * @return {*}
 */
void Matrix_Multiply(struct Matrix *input,struct Matrix *mulTerm,struct Matrix *output);

/**
 * @name: Matrix_Sum
 * @msg: Add the Matrix with adding term and return the output
 * @param {struct Matrix} *input              - (input)the matrix which need to process add
 * @param {struct Matrix} *addTerm            - (input)the addTerm which being add
 * @param {struct Matrix} *output             - (output)the matrix which store the solution
 * @return {*}
 */
void Matrix_Sum(struct Matrix *input,struct Matrix *addTerm,struct Matrix *output);

/**
 * @name: Matrix_ToZero
 * @msg: let the matrix feature all become zero
 * @param {struct Matrix} *input              - (output)the matrix which need to process to zero
 * @return {*}
 */
void Matrix_ToZero(struct Matrix *input);

#endif