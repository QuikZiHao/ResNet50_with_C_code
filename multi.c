#include"multi.h"
#include"backpropagation.h"
#include"pthread.h"

void* Thread_Convolution(void *args)
{   
    struct ConvolutionArgs *convolution_args = (struct ConvolutionArgs*)args;
    int channel = convolution_args->input->channelSize ;
    int outputSize = convolution_args->output->rowSize;
    int kernelSize = convolution_args->kernel[0]->rowSize ;
    int kernelAmt = convolution_args->output->channelSize;
    int stride = convolution_args->stride;
    int paddingAmt = convolution_args->paddingAmt;
    int maxRow = convolution_args->input->rowSize;
    int maxColumn = convolution_args->input->columnSize;
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        for(int j = 0 ; j < outputSize ; j++)
        {
            for(int k = 0 ; k < outputSize ; k++ )
            {
                float temp = 0; // store the temporary after convolution give the value back to the output
                for(int i = 0 ; i < channel ; i ++)
                {   
                    for(int y = 0 ; y < kernelSize ; y ++)
                    {
                        for(int z = 0 ; z < kernelSize ; z++ )
                        {
                            int rowLoc = j*stride+y-paddingAmt;
                            int columnLoc = k*stride+z-paddingAmt;
                            //calculate the column is padding or not
                            if ((rowLoc < 0 || rowLoc >= maxRow ) || (columnLoc < 0 || columnLoc >= maxColumn ))
                            {
                                //use to ignore the padding term
                                continue;
                            }
                            temp += (convolution_args->input->feature[i][rowLoc][columnLoc] * convolution_args->kernel[num]->feature[i][y][z]);
                            //printf("%f",temp);
                        }
                    }
                }
                convolution_args->output->feature[num][j][k] = temp;
                //printf("before :%f",convolution_args->output->feature[num][j][k]);
                //printf(" after :%f\n",convolution_args->output->feature[num][j][k]);
            }
        }
    }
    free(args);
    return NULL;   
}

void Multi_Convolution(struct Matrix **input,struct Matrix **kernel,struct Matrix **output ,int paddingAmt,int stride)
{
    for(int num = 0 ; num < BATCH ; num = num + THREADSIZE)
    {
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            CONV_Args[i].input = input[num + i];
            CONV_Args[i].kernel = kernel;
            CONV_Args[i].output = output[num + i];
            CONV_Args[i].paddingAmt = paddingAmt;
            CONV_Args[i].stride = stride;
            if (pthread_create(&th[i], NULL, Thread_Convolution, &CONV_Args[i]) != 0) 
            {
                perror("pthread_create");
                exit(EXIT_FAILURE);
            }   
        }
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            if (pthread_join(th[i], NULL) != 0) 
            {
                perror("pthread_join");
                exit(EXIT_FAILURE);
            }
        }
    } 
}

/*
void* Thread_ReLU(void *args)
{
    struct Args *relu_args = (struct Args*)args;
    int channel = relu_args->input->channelSize;
    int size = relu_args->input->rowSize;
    for(int i = 0 ; i < channel ; i++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0; k < size ; k++)
            {
                if(relu_args->input->feature[i][j][k] < 0)
                {
                    relu_args->output->feature[i][j][k] = 0 ;
                }
                else
                {
                    relu_args->output->feature[i][j][k] = relu_args->input->feature[i][j][k] ;
                }
            }
        }
    }
    free(args);
}

void Multi_ReLU (struct Matrix **input,struct Matrix **output)
{
    for(int num = 0 ; num < BATCH ; num = num + THREADSIZE)
    {
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            matrixArgs[i].input = input[num + i];
            matrixArgs[i].output = output[num + i];
            if (pthread_create(&th[i], NULL, Thread_ReLU, &matrixArgs[i]) != 0) 
            {
                perror("pthread_create");
                exit(EXIT_FAILURE);
            }   
        }
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            if (pthread_join(th[i], NULL) != 0) 
            {
                perror("pthread_join");
                exit(EXIT_FAILURE);
            }
        }
    } 
}
*/

void* Thread_MaxPooLing(void *args)
{   
    struct MaxArgs *max_args = (struct MaxArgs*)args;
    int channel = max_args->output->channelSize;
    int size = max_args->output->rowSize;
    for(int i = 0 ; i < channel ; i ++)
    {   
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++ )
            {
                float maximun = -FLT_MAX; // initial with minimun value of float
                for(int y = 0 ; y < max_args->poolingSize ; y ++)
                {
                    for(int z = 0 ; z < max_args->poolingSize ; z++ )
                    {
                        int rowLoc = j*max_args->stride+y-max_args->paddingAmt;
                        int columnLoc = k*max_args->stride+z-max_args->paddingAmt;
                        //calculate the column is padding or not
                        if ((rowLoc < 0 || rowLoc >= max_args->input->rowSize ) || (columnLoc < 0 || columnLoc >= max_args->input->columnSize))
                        {
                            if(maximun < 0)
                            {
                                maximun = 0;
                            }
                            continue;
                        }
                        if(max_args->input->feature[i][rowLoc][columnLoc] > maximun)
                        {
                            maximun = max_args->input->feature[i][rowLoc][columnLoc];
                        }
                    }
                }
                max_args->output->feature[i][j][k] = maximun;
            }
        }
    }
    free(args);
    return NULL;   
}

void Multi_MaxPooLing(struct Matrix **input,struct Matrix **output,int poolingSize,int paddingAmt,int stride)
{
    for(int num = 0 ; num < BATCH ; num = num + THREADSIZE)
    {
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            Max_Args[i].input = input[num + i];
            Max_Args[i].output = output[num + i];
            Max_Args[i].poolingSize = poolingSize;
            Max_Args[i].paddingAmt = paddingAmt;
            Max_Args[i].stride = stride;
            if (pthread_create(&th[i], NULL, Thread_MaxPooLing, &Max_Args[i]) != 0) 
            {
                perror("pthread_create");
                exit(EXIT_FAILURE);
            }   
        }
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            if (pthread_join(th[i], NULL) != 0) 
            {
                perror("pthread_join");
                exit(EXIT_FAILURE);
            }
        }
    } 
}

/*
void* Thread_ToZero(void *args)
{
    struct Args *gradientArgs = (struct Args*)args;
    int channel = gradientArgs->input->channelSize;
    int size = gradientArgs->input->rowSize;
    for(int i = 0 ; i < channel ; i++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0; k < size ; k++)
            {
                gradientArgs->output->feature[i][j][k] = 0 ;
            }
        }
    }
    free(args);
}

void Multi_ToZero (struct Matrix **input)
{
    for(int num = 0 ; num < BATCH ; num = num + THREADSIZE)
    {
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            matrixArgs[i].input = input[num + i];
            matrixArgs[i].output = input[num + i];
            if (pthread_create(&th[i], NULL, Thread_ToZero, &matrixArgs[i]) != 0) 
            {
                perror("pthread_create");
                exit(EXIT_FAILURE);
            }   
        }
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            if (pthread_join(th[i], NULL) != 0) 
            {
                perror("pthread_join");
                exit(EXIT_FAILURE);
            }
        }
    } 
}
*/

void* Thread_Convolution_Variable(void *args)
{
    //input = lastterm gradient
    //output = gradient
    struct ConvolutionArgs *convolution_args = (struct ConvolutionArgs*)args;
    int channel = convolution_args->output->channelSize ;
    int size = convolution_args->input->rowSize;
    int kernelSize = convolution_args->kernel[0]->rowSize ;
    int kernelAmt = convolution_args->input->channelSize; 
    int stride = convolution_args->stride;
    int paddingAmt = convolution_args->paddingAmt;
    int maxSize = convolution_args->output->rowSize;
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            for(int k = 0 ; k < size ; k++ )
            {
                for(int i = 0 ; i < channel ; i ++)
                {   
                    for(int y = 0 ; y < kernelSize ; y ++)
                    {
                        for(int z = 0 ; z < kernelSize ; z++ )
                        {
                            int rowLoc = j*stride+y-paddingAmt;
                            int columnLoc = k*stride+z-paddingAmt;
                            //calculate the column is padding or not
                            if ((rowLoc < 0 || rowLoc >= maxSize ) || (columnLoc < 0 || columnLoc >= maxSize ))
                            {
                                //use to ignore the padding term
                                continue;
                            }
                            //dL/dX = dL/dy * dy/dx (only the kernel which sliding and fix the input will affect the gradient )
                            convolution_args->output->feature[i][rowLoc][columnLoc] += (convolution_args->input->feature[num][j][k] *
                                                                         convolution_args->kernel[num]->feature[i][y][z]);
                        }
                    }
                }
            }
        }
    } 
    free(args);  
}

void Multi_GradientConvolution(struct Matrix **lastTermGradient ,struct Matrix **kernel ,struct Matrix **gradient,
                           struct Matrix **variable ,struct Matrix **gradientKernel, int stride, int paddingAmt)
{
    int kernelAmt = lastTermGradient[0]->channelSize;
    for(int num = 0 ; num < BATCH ; num = num + THREADSIZE)
    {
        for(int i = 0 ; i < THREADSIZE ; i++)
        {
            CONV_Args[i].input = lastTermGradient[num + i];
            CONV_Args[i].kernel = kernel;
            Matrix_ToZero(gradient[num + i]);
            CONV_Args[i].output = gradient[num + i];
            CONV_Args[i].paddingAmt = paddingAmt;
            CONV_Args[i].stride = stride;
            if (pthread_create(&th[i], NULL,Thread_Convolution_Variable, &CONV_Args[i]) != 0) 
            {
                perror("pthread_create");
                exit(EXIT_FAILURE);
            }   
        }
        for(int i = 0 ; i < THREADSIZE/2 ; i++)
        {
            if (pthread_join(th[i], NULL) != 0) 
            {
                perror("pthread_join");
                exit(EXIT_FAILURE);
            }
        }
    } 
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        Matrix_ToZero(gradientKernel[num]);
    }
    Gradient_Convolution_Kernel(lastTermGradient,gradientKernel,variable,stride,paddingAmt);
    for(int num = 0 ; num < kernelAmt ; num ++)
    {
        Back_Descent(kernel[num],gradientKernel[num]);
    }
}