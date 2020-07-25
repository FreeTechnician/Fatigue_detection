from face_decide.net import MainNet
import thop,torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trunk = MainNet().to(device)
x = torch.randn([1,3,128,128],dtype=torch.float32).to(device)

start_time = time.time()
y = trunk(x)
end_time = time.time()
t_pnet = end_time - start_time
print(t_pnet)


flops, params = thop.profile(trunk,(x,))
print(flops/1024/1024, params*4/1024/1024)