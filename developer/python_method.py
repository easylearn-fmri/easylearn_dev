class cal:
    cal_name = 'computer'
    def __init__(self,x,y):
        self.x = x
        self.y = y

    #在cal_add函数前加上@property，使得该函数可直接调用，封装起来   
    @property          
    def cal_add(self):
        return self.x + self.y

    #在cal_info函数前加上@classmethon，则该函数变为类方法，该函数只能访问到类的数据属性，不能获取实例的数据属性
    @classmethod       
    def cal_info(cls):  #python自动传入位置参数cls就是类本身
        print(cls.cal_name)   # cls.cal_name调用类自己的数据属性

    @staticmethod       #静态方法 类或实例均可调用
    def cal_test(a,b,c): #改静态方法函数里不传入self 或 cls
        print(a,b,c)


c1 = cal(20,11)
c1.cal_test(1,2,3)
c1.cal_info()
print(c1.cal_add)