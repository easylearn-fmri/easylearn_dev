# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:13:01 2019

@author: lenovo
"""

import fmriprep_docker
import docker
client = docker.from_env()
client.images.get("python")
client.images.get("cgyan/dpabi")

# 运行
client.containers.run("python:latest", "")
client.containers.run("ubuntu:latest", "echo hello")
container = client.containers.run("python", "", detach=True)

container.start()
print(client.containers.list())

# 打印容器
for container in client.containers.list():
    print(container.id)
    container.stop()

# 获取所有镜像
for image in client.images.list():
    print(image.id)

# 拉镜像
#image = client.images.pull("alpine")
# print(image.id)

# docker run - ti - -rm - v F: / Docker / data: / data: ro - v F: / Docker / data: / out poldracklab / fmriprep / data / out / out participant
# docker run poldracklab / fmriprep F: \Docker\data F: \Docker\data participant

# docker run poldracklab / fmriprep data data participant
