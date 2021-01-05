# 1. 进入你folk的easylearn本地文件夹
cd "path/to/your/folked_easylearn"
# 2. 查看关联情况
git remote -v  
# 3. 如果没有关联，则关联到easylearn母仓库
git remote add upstream git@github.com:easylearn-fmri/easylearn_dev.git
# 4. 再次查看关联情况
git remote -v  
# 5. 如果关联成功，则可以获取母仓库源代码
git fetch upstream  
# 6. merge到自己项目的dev分支（或者其它分支）
git merge upstream/dev  
# 7. 最后把最新的代码推送到你的github的dev分支（或其它分支）
git push origin dev
# 8. 如果你对代码做了贡献，并希望给update_stream（即easylearn）发送Pull Request
 >打开你自己的github网址
 >点击Pull Request -> 点击New Pull Request -> 输入Title和功能说明 -> 点击Send pull request
 > 注意：Pull Request 到目仓库的dev分支。即不要Pull Request 到master/main 分支