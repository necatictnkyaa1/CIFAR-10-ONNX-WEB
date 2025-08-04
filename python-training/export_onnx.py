import torch
import torch.onnx
from train_model import CIFAR10CNN
import numpy as np

def export_to_onnx():
    device=torch.device("cpu")
    
    model=CIFAR10CNN()
    model.load_state_dict('../models/cifar10_cnn.pth',map_location=device)
    
    model.eval()
    
    #webde tek resim tahmin edeceğimiz için batch_size 1
    dummy_input=torch.randn(batch_size=1,channelse=3,height=32,width=32)
    
    onnx_path='../models/cifar10_model.onnx'
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        
        # - Model weights'lerini ONNX dosyasına gömme
        # - False olursa sadece architecture kaydedilir
        export_params=True,
        
        #11 Stabil, çoğu operator destekleniyor bu yüzden bunu kullanacağız
        #9 Eski ama uyumlu
        #13+ Yeni ama support riski
        opset_version=11,
        
        do_constant_folding=True,
        
        #JavaScript'te tensor'lara isimle erişeceğiz
        #'input': session.run({'input': tensor}) şeklinde
        #'output': results.output.data
        input_names=['input'],          
        output_names=['output'],
        
        dynamic_axes={
            'input': {0: 'batch_size'},    # Input'un 0. boyutu değişken
            'output': {0: 'batch_size'}    # Output'un 0. boyutu değişken
        }
    )
    