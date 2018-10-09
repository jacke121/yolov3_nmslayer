import os
import pynvml

pynvml.nvmlInit()


def usegpu(need_gpu_count=1):
    nouse = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        # 这里的0是GPU id
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / meminfo.total
        if used < 0.5:
            nouse.append(index)
    if len(nouse) >= need_gpu_count:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, nouse[:need_gpu_count]))
        print("use gpu", ','.join(map(str, nouse[:need_gpu_count])))
        return nouse[:need_gpu_count]
    elif len(nouse) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, nouse))
        print("use gpu",','.join(map(str, nouse)))
        return len(nouse)
    else:
        return 0


if __name__ == '__main__':

    gpus = usegpu(need_gpu_count=2)

    if gpus:
        print("use gpu ok")
    else:
        print("no gpu is valid")
