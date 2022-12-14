import torch
import numpy

# print(torch.__version__)
# print(torch.cuda.is_available())

"""创建空矩阵（张量）"""
x = torch.empty(5,3)     #5行3列

"""创建随机矩阵"""
y = torch.rand(5,3)

"""创建全零矩阵"""
z = torch.zeros(5,3,dtype=torch.long)

"""直接传入数据"""
x = torch.tensor([5.5,6.6])

"""返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充"""
y = y.new_ones(5,3,dtype=torch.double)
y = torch.randn_like(y,dtype=torch.float)

"""展示大小"""
print(y.size())

"""运算"""
x = torch.rand(5,3)
y = torch.rand(5,3)
print(x,y,x+y,torch.add(x,y))

"""索引"""
x = x[:,1]

"""view"""
x = torch.randn(2,8)
y = x.view(16)
z = x.view(-1,8)    #负号表示自动计算，1表示第二行8个元素
print(x,y,z)

"""与numpy的协同操作"""
a = torch.ones(5)
b = a.numpy()        #torch转numpy

a = numpy.ones(5)
b = torch.from_numpy(a)       #numpy转torch
