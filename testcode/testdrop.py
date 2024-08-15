#用户输入三个浮点数a、b、c
a = float(input())
b = float(input())
c = float(input())
#计算平均波动率
bol = ((a-b)/a+(b-c)/b)/2
#计算平均值
ave = (a+b+c)/3

print('%.2f' % bol)
print('%.2f' % ave)