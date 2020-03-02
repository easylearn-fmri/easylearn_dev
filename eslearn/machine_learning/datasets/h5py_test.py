# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:12:14 2018

@author: lenovo
"""
#创建
import h5py
#要是读取文件的话，就把w换成r
f=h5py.File("myh5py.hdf5","w")
d1=f.create_dataset("dset1", (20,), 'i')
for key in f.keys():
    print(key)
    print(f[key].name)
    print(f[key].shape)
    print(f[key].value)
    
## 赋值
import h5py
import numpy as np
f=h5py.File("myh5py.hdf5","w")
d1=f.create_dataset("dset1",(20,),'i')
#赋值
d1[...]=np.arange(20)
#或者我们可以直接按照下面的方式创建数据集并赋值
f["dset2"]=np.arange(15)

for key in f.keys():
    print(f[key].name)
    print(f[key].value)
    
# 已有numpy数组时
import h5py
import numpy as np
f=h5py.File("myh5py.hdf5","w")
a=np.arange(20)
d1=f.create_dataset("dset3",data=a)
for key in f.keys():
    print(f[key].name)
    print(f[key].value)

# 创建group
import h5py
import numpy as np
f=h5py.File("myh5py.hdf5","w")

#创建一个名字为bar的组
g1=f.create_group("bar")

#在bar这个组里面分别创建name为dset1,dset2的数据集并赋值。
g1["dset1"]=np.arange(10)
g1["dset2"]=np.arange(12).reshape((3,4))

for key in g1.keys():
    print(g1[key].name)
    print(g1[key].value)
    
# group and datasets
import h5py
import numpy as np
f=h5py.File("myh5py.hdf5","w")

#创建组bar1,组bar2，数据集dset
g1=f.create_group("bar1")
g2=f.create_group("bar2")
d=f.create_dataset("dset",data=np.arange(10))

#在bar1组里面创建一个组car1和一个数据集dset1。
c1=g1.create_group("car1")
d1=g1.create_dataset("dset1",data=np.arange(10))

#在bar2组里面创建一个组car2和一个数据集dset2
c2=g2.create_group("car2")
d2=g2.create_dataset("dset2",data=np.arange(10))

#根目录下的组和数据集
print(".............")
for key in f.keys():
    print(f[key].name)

#bar1这个组下面的组和数据集
print(".............")
for key in g1.keys():
    print(g1[key].name)


#bar2这个组下面的组和数据集
print(".............")
for key in g2.keys():
    print(g2[key].name)

#顺便看下car1组和car2组下面都有什么，估计你都猜到了为空。
print(".............")
print(c1.keys())
print(c2.keys())