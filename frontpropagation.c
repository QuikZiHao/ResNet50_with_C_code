#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "batchnorm.h"
#include "matrix.h"
#include "frontpropagation.h"
#include "init.h"


void Front_ReLU (struct Matrix *input,struct Matrix *output)
{
    int channel = input->channelSize;
    int size = input->rowSize;
    for(int i = 0 ; i < channel ; i++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0; k < size ; k++)
            {
                if(input->feature[i][j][k] < 0)
                {
                    output->feature[i][j][k] = 0;//-0.1 * input->feature[i][j][k] ;
                }
                else
                {
                    output->feature[i][j][k] = input->feature[i][j][k] ;
                }
            }
        }
    }
}

void FrontBatch_ReLU (struct Matrix **input,struct Matrix **output)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        Front_ReLU(input[num],output[num]);
    }
}

void Front_MaxPooLing(struct Matrix *input,struct Matrix *output,int poolingSize,int paddingAmt,int stride)
{
    int channel = output->channelSize;
    int size = output->rowSize;

    for(int i = 0 ; i < channel ; i ++)
    {   
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++ )
            {
                float maximun = -9999; // initial with minimun value of float
                for(int y = 0 ; y < poolingSize ; y ++)
                {
                    for(int z = 0 ; z < poolingSize ; z++ )
                    {
                        int rowLoc = j*stride+y-paddingAmt;
                        int columnLoc = k*stride+z-paddingAmt;
                        //calculate the column is padding or not
                        if ((rowLoc < 0 || rowLoc >= input->rowSize ) || (columnLoc < 0 || columnLoc >= input->columnSize))
                        {
                            if(maximun < 0)
                            {
                                maximun = 0;
                            }
                            continue;
                        }
                        if(input->feature[i][rowLoc][columnLoc] > maximun)
                        {
                            maximun = input->feature[i][rowLoc][columnLoc];
                        }
                    }
                }
                output->feature[i][j][k] = maximun;
            }
        }
    }
}

void FrontBatch_MaxPooLing(struct Matrix **input,struct Matrix **output,int poolingSize,int paddingAmt,int stride)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        Front_MaxPooLing(input[num],output[num],poolingSize,paddingAmt,stride);
    }
}

void FrontBatch_Convolution(struct Matrix **input,struct Matrix **kernel,struct Matrix **output ,int paddingAmt,int stride)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        Matrix_Convolution(input[num],kernel,output[num],paddingAmt,stride); 
    }
} 

void Front_GlobalAverage(struct Matrix *input,struct Matrix *output)
{
    int channel = input->channelSize;
    int size = input->rowSize;
    int featureSize = size*size*1.0;
    for(int i = 0 ; i < channel ; i++)
    {
        float ans = 0 ;
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++)
            {
                ans += input->feature[i][j][k];
            }
        }
        output->feature[0][i][0] = ans/featureSize;
    }
}

void FrontBatch_GlobalAverage(struct Matrix **input,struct Matrix **output)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        Front_GlobalAverage(input[num],output[num]);
    }
}

void Front_FullConnect(struct Matrix *input,struct Matrix *weight,struct Matrix *output,struct Matrix *bias)
{
    //weight = (1 X predictSize X rowSize)
    //input =  (1 X rowSize X 1)
    //output = (1 X PredictSize X 1)
    //bias = (1 X predictSize X 1)
    Matrix_Multiply(weight,input,output);
    Matrix_Sum(output,bias,output);     
}

void FrontBatch_FUllConnct(struct Matrix **input,struct Matrix *weight,struct Matrix **output,struct Matrix *bias)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        Front_FullConnect(input[num],weight,output[num],bias);
    }
}

void Front_Softmax(struct Matrix *input,int maximun,struct Matrix *output)
{
    int inputSize = input->rowSize;
    double summation = 0 ;
    for(int j = 0 ; j < inputSize ; j++)
    {
        output->feature[0][j][0] = exp(input->feature[0][j][0]-input->feature[0][maximun][0]);
        summation += output->feature[0][j][0];
    }
    for(int j = 0 ; j < inputSize ; j++)
    {
        output->feature[0][j][0] = output->feature[0][j][0]/summation; //change to percentage
    }
}

void FrontBatch_Softmax(struct Matrix **input,int *maximun,struct Matrix **output)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        Front_Softmax(input[num],maximun[num],output[num]);
    }
}

int Front_Predict(struct Matrix *input)
{
    float maximun = 0 ;
    int index = 0;
    int row = input->rowSize;
    for(int j = 0 ; j < row ; j ++)
    {
        if(input->feature[0][j][0] > maximun)
        {
            maximun = input->feature[0][j][0];
            index = j;
        }
    }
    return index;
}

void FrontBatch_Predict(struct Matrix **input, int *output)
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        output[num] = Front_Predict(input[num]);
    }
}

void Res_Sum(struct Matrix *input , struct Matrix *lastInput, struct Matrix *output)
{
    Matrix_Sum(input,lastInput,output);
}

void ResBatch_Sum(struct Matrix **input , struct Matrix **lastInput, struct Matrix **output)
{
    for(int num = 0 ; num < BATCH ; num ++)
    {
        Matrix_Sum(input[num],lastInput[num],output[num]);
    }
}

float FrontBatch_Accurancy(int *input,int *testCaseLabel,int index)
{
    int correct = 0;
    int start = index*BATCH;
    for(int num = 0 ; num < BATCH ; num++)
    {
        //Matrix_Print(feature_SoftMax[num]);
        //printf("num : %d predict : %d ,answer : %d \n",start+num,input[num],testCaseLabel[start+num]);
        if((int)input[num] == testCaseLabel[start+num])
        {
            correct++;
        }
    }
    float ans = correct * 1.000 / BATCH;
    return ans;
}

float FrontBatch_LossFunction(struct Matrix **input,int *testCaseLabel,int index)
{
    float loss = 0;
    int start = index*BATCH;
    for(int num = 0 ; num < BATCH ; num++)
    {
        loss = loss  + log(input[num]->feature[0][testCaseLabel[start+num]][0]);  
    }
    loss = -loss/BATCH;

    return loss;
}

