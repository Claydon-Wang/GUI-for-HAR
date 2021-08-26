from dataset.data_proc import *
import torch
from Layer_CNN import CNN
from Layer_MLP import MLP_WISDM, MixerBlock, FeedForward
import torch.nn as nn
from utils.model_profiling_base import *
# from utils.model_profiling_base import *
from train_test.train_test_proc import *

if __name__ == "__main__":  # 这个工程里的model_profiling未调用gpu
    test_loss_list = []
    test_acc_list = []
    test_time_elapsed_list = []

    # test_x_list = "./dataset/UCI/x_test.npy"
    # test_y_list = "./dataset/UCI/y_test.npy"
    #
    # data_test = HAR(test_x_list, test_y_list)
    # har_test_tensor = data_test.HAR_data()
    #
    # test_loader = Data.DataLoader(dataset=har_test_tensor, batch_size=1, shuffle=True, num_workers=5, )


    model_gMLP = torch.load('./model_save/WISDM/WISDM_gMLP_model.pth', map_location='cpu')
    model_gMLP = model_gMLP.to(torch.device("cpu"))

    model_CNN = torch.load('./model_save/WISDM/WISDM_CNN_model.pth', map_location='cpu')
    model_CNN = model_CNN.to(torch.device("cpu"))

    model_MLP = torch.load('./model_save/WISDM/WISDM_MLP_model.pth', map_location='cpu')
    model_MLP = model_MLP.to(torch.device("cpu"))



    profiling(model_gMLP, 'cpu', [200, 3],  # WISDM
              1, [1],
              True)
    tensor = (torch.rand(1, 1, 200, 3))


    print('                   MLP_TIME                       ')
    print('***************************************************')
    # torch.cuda.synchronize()
    start = time.time()
    output_MLP = model_MLP(tensor)
    # torch.cuda.synchronize()
    end = time.time()
    Inference_time_MLP = (end-start)*1000
    print(Inference_time_MLP, 'ms')

    print('                   gMLP_TIME                       ')
    print('***************************************************')
    # torch.cuda.synchronize()
    start = time.time()
    output_gMLP = model_gMLP(tensor)
    # torch.cuda.synchronize()
    end = time.time()
    Inference_time_gMLP = (end-start)*1000
    print(Inference_time_gMLP, 'ms')

    print('                    CNN_TIME                       ')
    print('***************************************************')
    start = time.time()
    output_CNN = model_CNN(tensor)
    # torch.cuda.synchronize()
    end = time.time()
    Inference_time_CNN = (end-start)*1000
    print(Inference_time_CNN, 'ms')


   





   



