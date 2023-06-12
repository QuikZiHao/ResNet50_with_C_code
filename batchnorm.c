#include "matrix.h"
#include "batchnorm.h"

struct BN *BN_Init(int channel)
{
    struct BN *input = (struct BN *)malloc(sizeof(struct BN));
    input->channelSize = channel;
    input->beta =  (float *)malloc(channel * sizeof(float));
    input->gamma =  (float *)malloc(channel * sizeof(float));
    input->mean = (float *)malloc(channel * sizeof(float));
    input->variance =  (float *)malloc(channel * sizeof(float));
    input->runningMean = (float *)malloc(channel * sizeof(float));
    input->runningVar = (float *)malloc(channel * sizeof(float));
    for(int i = 0 ; i < channel ; i++)
    {
        input->mean[i] = 0;
        input->variance[i] = 0;
        input->beta[i] = 0;
        input->gamma[i] = 1.0;
        input->runningMean[i] = 0;
        input->runningVar[i] = 1.0;   
    }
    return input;
}

void *BN_GetCoeff(struct Matrix **input , struct BN *coeff)
{
    int channel = input[0]->channelSize;
    int size = input[0]->rowSize;
    int amount = BATCH * size * size ;     
    for(int i  = 0 ; i < channel ; i++)
    {
        coeff->mean[i] =0 ;
        coeff->variance[i] = 0;
        float temp =  0;
        for(int num = 0 ; num < BATCH ; num ++)
        {
            for (int j = 0 ; j < size ; j++)
            {
                for(int k = 0 ; k < size ; k++)
                {
                    temp = temp + input[num]->feature[i][j][k];
                    //printf("batch = %d i = %d j = %d k = %d temp = %.0f\n",batch,i,j,k,temp);
                }
            }
        }
        coeff->mean[i] = temp/amount ;

        coeff->runningMean[i]  = (1-MOMENTUM)*coeff->runningMean[i] + MOMENTUM*coeff->mean[i];
        //mean_new = (1-momentum)mean_old+momentum*mean this batch
    }
    
    for(int i  = 0 ; i < channel ; i++)
    {
        float temp =  0;
        float avg = coeff->mean[i];
        for(int batch = 0 ; batch < BATCH ; batch ++)
        {
            for (int j = 0 ; j < size ; j++)
            {
                for(int k = 0 ; k < size ; k++)
                {
                    temp = temp + pow(input[batch]->feature[i][j][k] - avg ,2);
                }
            }
        }
        coeff->variance[i] = temp/(amount);
        coeff->runningVar[i]  = (1-MOMENTUM)*coeff->runningVar[i] + MOMENTUM*temp/(amount-1);
    }
}

void BN_BatchNorm(struct Matrix *input , struct Matrix *output , struct BN *coeff )
{   
    int channel = input->channelSize;
    int size = input->rowSize;
    for(int i  = 0 ; i < channel ; i++)
    {
        float avg = coeff->mean[i];
        float std = coeff->variance[i]; 
        float gamma = coeff->gamma[i];
        float beta = coeff->beta[i];
        for (int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++)
            {
                float temp = input->feature[i][j][k] ;
                temp = gamma*(temp -avg)/pow(std+EPSILON,0.5) + beta;
                output->feature[i][j][k]  = temp;
            }
        }
    }
}

void **BNBatch_BatchNorm(struct Matrix **input ,struct Matrix **output, struct BN *coeff )
{
    for(int num = 0 ; num < BATCH ; num++)
    {
        BN_BatchNorm(input[num],output[num],coeff);
    }
}

