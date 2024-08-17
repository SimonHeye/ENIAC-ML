import matplotlib.pyplot as plt

#1、准备数据
x=[5,10,15]
ZETT=[21.49,17.16,12.78]
PCRED=[18.40,22.30,21.64]
ZS_SKA=[44.00,26.40,20.26]
AML_EIKC=[45.21,38.81,38.48]

#2、创建画布
plt.figure(dpi=100)

#3、绘制折线图
plt.plot(x,ZETT,label='ZETT',color='#EBA782',linestyle='-',linewidth=2)
plt.plot(x,PCRED,label='PCRED',color='#549F9A',linestyle='-',linewidth=2)
plt.plot(x,ZS_SKA,label='ZS_SKA',color='#4370B4',linestyle='-',linewidth=2)
plt.plot(x,AML_EIKC,label='AML_EIKC',color='#C30078',linestyle='-',linewidth=2)

plt.scatter(x, ZETT, color='#EBA782',s=20)
plt.scatter(x, PCRED, color='#549F9A',s=20)
plt.scatter(x, ZS_SKA, color='#4370B4',s=20)
plt.scatter(x, AML_EIKC, color='#C30078',s=20)

#4、添加辅助显示信息
#添加图例
plt.legend(fontsize=12)

#添加x,y轴刻度
_xtick_labels=['5','10','15']
plt.xticks(x,_xtick_labels, fontsize=14)

#添加x,y轴名称
plt.xlabel('Unseen Relations', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

#添加标题
#添加网格

#5、显示图像    
plt.show()