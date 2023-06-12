/*
 * @Descripttion: input the learning/test case to the program
 * @version:2.0
 * @Author: Quikziii
 * @email: quikziii@gmail.com
 * @Date: 2023-01-24 
 * @LastEditors: Quikziii
 * @LastEditTime: 2023-03-14
 */
#ifndef _INPUT_H
#define _INPUT_H

//size of per image
# define BIN 3072
//size of per file
#define TESTSIZE 50000

int predictSize; //record the size for cifar 10 = 10 , cifar 100 = 100
struct Matrix **input; //store the test case
int *testCaseLabel;     //store the answer of test case

/**
 * @name: ReadFile
 * @msg: read bin file to the program
 * @param {void} *func                - (input)select the BIN file mode
 * @param {{char} *filename            - (input)the filename of the learning/test case
 * @return {*}
 */
void ReadFile(void (*func)(struct Matrix **,int *),struct Matrix **input, int *testCaseLabel);

/**
 * @name: Cifar10
 * @msg: read Cifar10 bin file to the program
 * @param {{char} *fileName            - (input)the filename of the learning/test case
 * @return {*}
 */
void Cifar10(struct Matrix **input, int *testCaseLabel);

/**
 * @name: Cifar100
 * @msg: read Cifar100 bin file to the program
 * @param {char} *fileName            - (input)the filename of the learning/test case
 * @return {*}
 */
void Cifar100(struct Matrix **input, int *testCaseLabel);

/**
 * @name: InitInput
 * @msg: initial the matrix list of the input data
 * @param {struct Matrix **} **input    - (input)the struct list which need to be initial
 * @return {*}
 */
void InitInput(struct Matrix **input);

/**
 * @name: Write_Error
 * @msg: record the acc and loss per epochs
 * @param {char} *filename          - (input)the filename to record the experiment data
 * @param {float} *accurancy        - (input) accurancy of this epoch
 * @param {float} *loss             - (input) loss of this epoch
 * @return {*}
 */
void Write_Error(char *fileName , int iterateTime , float accurancy , float loss,double elapsed_time);


#endif