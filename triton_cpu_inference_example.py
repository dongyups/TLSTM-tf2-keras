import os
import psutil
import datetime, time
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np


### 수동 메모리 확인 ###
def get_process_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


### 입력 데이터 샘플 1 batch ###
input_data = np.array([[ 86.       ,   1.       ,   0.       ,  17.36     ,  63.       ,
        159.0441375, 100.       ,  76.       , 212.       ,  68.       ,
         60.       , 138.       ,  13.       ,   1.4      ,  16.       ,
         11.       ,  17.       ,   1.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   1.       ,   0.       ,
          1.       ,   0.       ,   0.       ,   1.       ,   1.       ,
          0.       ,   1.       ,   0.       ,   1.       ,   0.       ,
          0.       ,   1.       ,   1.       ,   0.       ,   1.       ,
          0.       ,   1.       ,   0.       ,   1.       ,   0.       ,
          1.       ,   0.       ,   0.       ,   0.       ,   1.       ,
          0.       ,   0.       ,   1.       ,   0.       ,   0.       ,
          0.       ,   1.       ,   0.       ,   0.       ],
       [ 88.       ,   1.       ,   0.       ,  17.01     ,  66.       ,
        110.       ,  60.       ,  87.       , 227.       , 131.       ,
         61.       , 140.       ,  12.8      ,   1.4      ,  28.       ,
         14.       ,  13.       ,   1.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   1.       ,   0.       ,
          1.       ,   0.       ,   0.       ,   1.       ,   1.       ,
          0.       ,   1.       ,   0.       ,   1.       ,   0.       ,
          0.       ,   1.       ,   1.       ,   0.       ,   1.       ,
          0.       ,   1.       ,   0.       ,   1.       ,   0.       ,
          1.       ,   0.       ,   0.       ,   0.       ,   1.       ,
          0.       ,   0.       ,   1.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   1.       ],
       [ 90.       ,   1.       ,   0.       ,  17.72     ,  81.       ,
        114.       ,  55.       ,  75.       , 215.       , 107.       ,
         45.       , 148.       ,  10.6      ,   1.6      ,  19.       ,
          8.       ,  12.       ,   1.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   1.       ,   0.       ,
          1.       ,   0.       ,   0.       ,   1.       ,   1.       ,
          0.       ,   1.       ,   0.       ,   1.       ,   0.       ,
          0.       ,   1.       ,   1.       ,   0.       ,   1.       ,
          0.       ,   1.       ,   0.       ,   1.       ,   0.       ,
          1.       ,   0.       ,   0.       ,   0.       ,   1.       ,
          0.       ,   0.       ,   1.       ,   0.       ,   0.       ,
          0.       ,   1.       ,   0.       ,   0.       ],
       [  0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ],
       [  0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ,   0.       ,
          0.       ,   0.       ,   0.       ,   0.       ]]).astype('float32')
input_time = np.array([  0., 614., 951.,   0.,   0.]).astype('float32')
input_position = np.array(2).astype('int32')

input_data = np.expand_dims(input_data, axis=0)
input_time = np.expand_dims(input_time, axis=0)
input_position = np.expand_dims(input_position, axis=0)


### triton 서버 설정 ###
'''
!docker run --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 
    -v /mnt/data2/dshin/triton-inference:/models nvcr.io/nvidia/tritonserver:23.02-py3 tritonserver 
    --model-repository=/models 
    --log-verbose 1 
    --strict-model-config=True 
    --gpus all
I0427 09:00:14.706186 1 http_server.cc:3477] Started HTTPService at 0.0.0.0:8000
I0427 09:00:14.705907 1 grpc_server.cc:4868] Started GRPCInferenceService at 0.0.0.0:8001
I0427 09:00:14.747476 1 http_server.cc:184] Started Metrics Service at 0.0.0.0:8002
'''
initial_memory_usage = get_process_memory_usage()
triton_client = grpcclient.InferenceServerClient(
    url='10.10.10.15:8001',
    ssl=None,
    root_certificates=None,
    private_key=None,
    certificate_chain=None,
    verbose=False,
)


### inputs/outputs 맵핑 ###
inputs = [
    grpcclient.InferInput("input_1", input_data.shape, np_to_triton_dtype(input_data.dtype)),
    grpcclient.InferInput("input_2", input_time.shape, np_to_triton_dtype(input_time.dtype)),
    grpcclient.InferInput("input_3", input_position.shape, np_to_triton_dtype(input_position.dtype)),
]
inputs[0].set_data_from_numpy(input_data)
inputs[1].set_data_from_numpy(input_time)
inputs[2].set_data_from_numpy(input_position)
outputs = [
    grpcclient.InferRequestedOutput("output_1")
]


### trition inference loop ###
i = 0
end_time = datetime.datetime.now() + datetime.timedelta(hours=24)

while datetime.datetime.now() < end_time:
    infer_time = time.time()

    response = triton_client.infer(
    model_name="my_model",
    model_version="1",
    inputs=inputs,
    outputs=outputs,
    )

    output_tensor = response.as_numpy('output_1')
    current_memory_usage = get_process_memory_usage() # 모델 사이즈 포함

    if i % 1000 == 0:
        print(f"Sample #{i} Inference Time: {int((time.time()-infer_time)*1000)}: ms \
              Total Memory usage: {current_memory_usage/ (1024 * 1024)} MB \
              Memory Leak amount: {(current_memory_usage - initial_memory_usage) / (1024 * 1024):.2f} MB \
              Result: {output_tensor}")
    i += 1

